#!/usr/bin/env python3
"""
Security Validation Script

Comprehensive security validation for the Telegram ML Bot application.
Verifies all critical security fixes are properly implemented.

Usage:
    python scripts/security_validation.py [--fix] [--report]
    
Options:
    --fix: Attempt to fix found issues automatically
    --report: Generate detailed security report
    --verbose: Show detailed output
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.secrets_manager import get_secrets_manager
from app.config.settings import get_settings
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SecurityIssue:
    """Represents a security issue found during validation."""
    
    def __init__(
        self, 
        severity: str, 
        category: str, 
        title: str, 
        description: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        fix_suggestion: Optional[str] = None
    ):
        self.severity = severity  # CRITICAL, HIGH, MEDIUM, LOW
        self.category = category
        self.title = title
        self.description = description
        self.file_path = file_path
        self.line_number = line_number
        self.fix_suggestion = fix_suggestion
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'severity': self.severity,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'fix_suggestion': self.fix_suggestion,
            'timestamp': self.timestamp.isoformat()
        }


class SecurityValidator:
    """Comprehensive security validation system."""
    
    def __init__(self, fix_issues: bool = False, verbose: bool = False):
        self.fix_issues = fix_issues
        self.verbose = verbose
        self.issues: List[SecurityIssue] = []
        self.project_root = Path(__file__).parent.parent
        self.settings = None
        self.secrets_manager = None
    
    async def initialize(self):
        """Initialize validator components."""
        try:
            self.settings = get_settings()
            self.secrets_manager = get_secrets_manager()
            logger.info("Security validator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize validator: {e}")
            raise
    
    def add_issue(self, issue: SecurityIssue):
        """Add a security issue to the list."""
        self.issues.append(issue)
        if self.verbose:
            logger.warning(
                f"[{issue.severity}] {issue.title}",
                category=issue.category,
                description=issue.description
            )
    
    def validate_authentication_implementation(self) -> bool:
        """Validate JWT authentication is properly implemented."""
        logger.info("Validating authentication implementation...")
        
        # Check if auth module exists
        auth_file = self.project_root / "app" / "core" / "auth.py"
        if not auth_file.exists():
            self.add_issue(SecurityIssue(
                "CRITICAL", "Authentication", "Missing authentication module",
                "The authentication module (app/core/auth.py) is missing",
                fix_suggestion="Create comprehensive authentication module with JWT support"
            ))
            return False
        
        # Check auth implementation
        try:
            with open(auth_file, 'r') as f:
                auth_content = f.read()
            
            # Check for required functions
            required_functions = [
                'get_current_user', 'require_permission', 'verify_token',
                'create_access_token', 'AuthenticatedUser'
            ]
            
            for func in required_functions:
                if func not in auth_content:
                    self.add_issue(SecurityIssue(
                        "HIGH", "Authentication", f"Missing {func} function",
                        f"Authentication module is missing {func} function",
                        fix_suggestion=f"Implement {func} function in auth module"
                    ))
            
            # Check for RBAC implementation
            if 'Permission' not in auth_content or 'UserRole' not in auth_content:
                self.add_issue(SecurityIssue(
                    "HIGH", "Authorization", "Missing RBAC implementation",
                    "Role-based access control system is not implemented",
                    fix_suggestion="Implement Permission and UserRole enums with proper mappings"
                ))
        
        except Exception as e:
            self.add_issue(SecurityIssue(
                "HIGH", "Authentication", "Cannot validate auth module",
                f"Error reading authentication module: {e}"
            ))
        
        return True
    
    def validate_endpoint_security(self) -> bool:
        """Validate API endpoints have proper authentication."""
        logger.info("Validating endpoint security...")
        
        # Check users.py endpoints
        users_file = self.project_root / "app" / "api" / "v1" / "users.py"
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    users_content = f.read()
                
                # Check for authentication decorators
                endpoints = re.findall(r'@router\.(get|post|put|delete)\([^)]*\)', users_content)
                authenticated_endpoints = len(re.findall(r'current_user.*Depends\(.*get_current_user', users_content))
                
                if len(endpoints) > authenticated_endpoints:
                    self.add_issue(SecurityIssue(
                        "CRITICAL", "Authorization", "Unauthenticated endpoints found",
                        f"Found {len(endpoints)} endpoints but only {authenticated_endpoints} require authentication",
                        fix_suggestion="Add authentication decorators to all sensitive endpoints"
                    ))
            
            except Exception as e:
                self.add_issue(SecurityIssue(
                    "MEDIUM", "Validation", "Cannot validate users endpoints",
                    f"Error reading users.py: {e}"
                ))
        
        return True
    
    def validate_secrets_management(self) -> bool:
        """Validate secrets are properly managed."""
        logger.info("Validating secrets management...")
        
        # Check if secrets manager exists
        secrets_file = self.project_root / "app" / "core" / "secrets_manager.py"
        if not secrets_file.exists():
            self.add_issue(SecurityIssue(
                "CRITICAL", "Secrets", "Missing secrets manager",
                "Production secrets manager is not implemented",
                fix_suggestion="Implement encrypted secrets management system"
            ))
            return False
        
        # Check .env file for hardcoded secrets
        env_file = self.project_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    env_content = f.read()
                
                # Look for patterns that suggest hardcoded secrets
                sensitive_patterns = [
                    r'(TOKEN|SECRET|KEY|PASSWORD)\s*=\s*[^\s#].*[a-zA-Z0-9]{10,}',
                    r'sk_[a-zA-Z0-9]+',  # Stripe secret keys
                    r'bot[0-9]+:[a-zA-Z0-9_-]+',  # Telegram bot tokens
                    r'eyJ[a-zA-Z0-9]+',  # JWT tokens
                ]
                
                for pattern in sensitive_patterns:
                    matches = re.findall(pattern, env_content, re.IGNORECASE)
                    if matches:
                        self.add_issue(SecurityIssue(
                            "CRITICAL", "Secrets", "Hardcoded secrets in .env",
                            f"Found potential hardcoded secrets: {len(matches)} matches",
                            fix_suggestion="Move secrets to encrypted storage using secrets manager"
                        ))
                        break
            
            except Exception as e:
                self.add_issue(SecurityIssue(
                    "MEDIUM", "Validation", "Cannot validate .env file",
                    f"Error reading .env: {e}"
                ))
        
        return True
    
    def validate_webhook_security(self) -> bool:
        """Validate webhook signature verification."""
        logger.info("Validating webhook security...")
        
        # Check main.py for webhook security
        main_file = self.project_root / "app" / "main.py"
        if main_file.exists():
            try:
                with open(main_file, 'r') as f:
                    main_content = f.read()
                
                # Check for webhook signature verification
                if 'verify_telegram_webhook' not in main_content:
                    self.add_issue(SecurityIssue(
                        "CRITICAL", "Webhooks", "Missing webhook signature verification",
                        "Telegram webhook does not verify signatures",
                        fix_suggestion="Implement webhook signature verification in telegram_webhook endpoint"
                    ))
                
                # Check for rate limiting on webhook
                if 'rate_limit_check' not in main_content:
                    self.add_issue(SecurityIssue(
                        "HIGH", "Webhooks", "Missing webhook rate limiting",
                        "Webhook endpoint lacks rate limiting protection",
                        fix_suggestion="Add rate limiting to webhook endpoint"
                    ))
            
            except Exception as e:
                self.add_issue(SecurityIssue(
                    "MEDIUM", "Validation", "Cannot validate main.py",
                    f"Error reading main.py: {e}"
                ))
        
        return True
    
    def validate_input_validation(self) -> bool:
        """Validate input validation and sanitization."""
        logger.info("Validating input validation...")
        
        # Check if input validation middleware exists
        validation_file = self.project_root / "app" / "middleware" / "input_validation.py"
        if not validation_file.exists():
            self.add_issue(SecurityIssue(
                "HIGH", "Input Validation", "Missing input validation middleware",
                "Input validation and sanitization middleware is missing",
                fix_suggestion="Implement comprehensive input validation middleware"
            ))
            return False
        
        # Check main.py for middleware usage
        main_file = self.project_root / "app" / "main.py"
        if main_file.exists():
            try:
                with open(main_file, 'r') as f:
                    main_content = f.read()
                
                if 'InputValidationMiddleware' not in main_content:
                    self.add_issue(SecurityIssue(
                        "HIGH", "Input Validation", "Input validation middleware not enabled",
                        "Input validation middleware is not added to the application",
                        fix_suggestion="Add InputValidationMiddleware to main.py"
                    ))
            
            except Exception as e:
                self.add_issue(SecurityIssue(
                    "MEDIUM", "Validation", "Cannot validate middleware usage",
                    f"Error validating middleware: {e}"
                ))
        
        return True
    
    def validate_cors_configuration(self) -> bool:
        """Validate CORS configuration is secure."""
        logger.info("Validating CORS configuration...")
        
        try:
            # Check CORS origins in settings
            cors_origins = self.settings.security.cors_origins
            
            # Check for overly permissive CORS
            if '*' in cors_origins:
                self.add_issue(SecurityIssue(
                    "HIGH", "CORS", "Overly permissive CORS configuration",
                    "CORS allows all origins (*) which is insecure for production",
                    fix_suggestion="Specify exact allowed origins in CORS_ORIGINS"
                ))
            
            # Check for localhost in production
            if self.settings.is_production:
                localhost_origins = [origin for origin in cors_origins if 'localhost' in origin]
                if localhost_origins:
                    self.add_issue(SecurityIssue(
                        "MEDIUM", "CORS", "Localhost origins in production",
                        f"Production environment has localhost origins: {localhost_origins}",
                        fix_suggestion="Remove localhost origins from production CORS configuration"
                    ))
        
        except Exception as e:
            self.add_issue(SecurityIssue(
                "MEDIUM", "Validation", "Cannot validate CORS configuration",
                f"Error validating CORS: {e}"
            ))
        
        return True
    
    def validate_security_headers(self) -> bool:
        """Validate security headers are implemented."""
        logger.info("Validating security headers...")
        
        # Check if security headers middleware exists
        headers_file = self.project_root / "app" / "middleware" / "security_headers.py"
        if not headers_file.exists():
            self.add_issue(SecurityIssue(
                "MEDIUM", "Security Headers", "Missing security headers middleware",
                "Security headers middleware is not implemented",
                fix_suggestion="Implement SecurityHeadersMiddleware with CSP, HSTS, etc."
            ))
            return False
        
        # Check for proper header implementation
        try:
            with open(headers_file, 'r') as f:
                headers_content = f.read()
            
            required_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options', 
                'X-XSS-Protection',
                'Content-Security-Policy',
                'Referrer-Policy'
            ]
            
            for header in required_headers:
                if header not in headers_content:
                    self.add_issue(SecurityIssue(
                        "MEDIUM", "Security Headers", f"Missing {header} header",
                        f"Security headers middleware does not set {header}",
                        fix_suggestion=f"Add {header} header to security middleware"
                    ))
        
        except Exception as e:
            self.add_issue(SecurityIssue(
                "MEDIUM", "Validation", "Cannot validate security headers",
                f"Error validating headers: {e}"
            ))
        
        return True
    
    def validate_session_management(self) -> bool:
        """Validate session management implementation."""
        logger.info("Validating session management...")
        
        # Check auth module for session management
        auth_file = self.project_root / "app" / "core" / "auth.py"
        if auth_file.exists():
            try:
                with open(auth_file, 'r') as f:
                    auth_content = f.read()
                
                session_functions = [
                    'create_session', 'get_session', 'invalidate_session'
                ]
                
                for func in session_functions:
                    if func not in auth_content:
                        self.add_issue(SecurityIssue(
                            "HIGH", "Session Management", f"Missing {func} function",
                            f"Session management is missing {func} function",
                            fix_suggestion=f"Implement {func} in authentication module"
                        ))
                
                # Check for Redis session storage
                if 'redis' not in auth_content.lower():
                    self.add_issue(SecurityIssue(
                        "MEDIUM", "Session Management", "No Redis session storage",
                        "Sessions should be stored in Redis for scalability",
                        fix_suggestion="Implement Redis-based session storage"
                    ))
            
            except Exception as e:
                self.add_issue(SecurityIssue(
                    "MEDIUM", "Validation", "Cannot validate session management",
                    f"Error validating sessions: {e}"
                ))
        
        return True
    
    def validate_gdpr_compliance(self) -> bool:
        """Validate GDPR compliance implementation."""
        logger.info("Validating GDPR compliance...")
        
        # Check if GDPR endpoints exist
        gdpr_file = self.project_root / "app" / "api" / "v1" / "gdpr.py"
        if not gdpr_file.exists():
            self.add_issue(SecurityIssue(
                "HIGH", "GDPR", "Missing GDPR compliance endpoints",
                "GDPR compliance endpoints are not implemented",
                fix_suggestion="Implement GDPR endpoints for data export, deletion, and consent management"
            ))
            return False
        
        # Check for required GDPR endpoints
        try:
            with open(gdpr_file, 'r') as f:
                gdpr_content = f.read()
            
            required_endpoints = [
                'export_user_data', 'delete_user_data', 'get_consent_status',
                'update_consent', 'get_privacy_policy'
            ]
            
            for endpoint in required_endpoints:
                if endpoint not in gdpr_content:
                    self.add_issue(SecurityIssue(
                        "MEDIUM", "GDPR", f"Missing {endpoint} endpoint",
                        f"GDPR compliance is missing {endpoint} endpoint",
                        fix_suggestion=f"Implement {endpoint} for GDPR compliance"
                    ))
        
        except Exception as e:
            self.add_issue(SecurityIssue(
                "MEDIUM", "Validation", "Cannot validate GDPR implementation",
                f"Error validating GDPR: {e}"
            ))
        
        return True
    
    async def validate_secrets_in_storage(self) -> bool:
        """Validate required secrets are available."""
        logger.info("Validating secrets availability...")
        
        try:
            validation_results = await self.secrets_manager.validate_required_secrets()
            
            for secret_name, is_available in validation_results.items():
                if not is_available:
                    self.add_issue(SecurityIssue(
                        "CRITICAL", "Secrets", f"Missing required secret: {secret_name}",
                        f"Required secret {secret_name} is not available in secrets storage",
                        fix_suggestion=f"Set {secret_name} in encrypted secrets storage"
                    ))
        
        except Exception as e:
            self.add_issue(SecurityIssue(
                "HIGH", "Secrets", "Cannot validate secrets storage",
                f"Error validating secrets: {e}",
                fix_suggestion="Ensure secrets manager is properly configured"
            ))
        
        return True
    
    def validate_dependency_security(self) -> bool:
        """Check for known security vulnerabilities in dependencies."""
        logger.info("Validating dependency security...")
        
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                # Run safety check if available
                result = subprocess.run(
                    ['python', '-m', 'pip', 'list', '--format=json'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    # Basic dependency validation
                    packages = json.loads(result.stdout)
                    
                    # Check for outdated critical packages
                    critical_packages = {
                        'fastapi': '0.100.0',
                        'pydantic': '1.10.0',
                        'sqlalchemy': '1.4.0',
                        'cryptography': '3.4.0'
                    }
                    
                    for package in packages:
                        name = package['name'].lower()
                        version = package['version']
                        
                        if name in critical_packages:
                            # Simple version comparison (improve as needed)
                            if version < critical_packages[name]:
                                self.add_issue(SecurityIssue(
                                    "MEDIUM", "Dependencies", f"Outdated package: {name}",
                                    f"Package {name} version {version} may have security vulnerabilities",
                                    fix_suggestion=f"Update {name} to version >= {critical_packages[name]}"
                                ))
            
            except Exception as e:
                logger.warning(f"Could not validate dependencies: {e}")
        
        return True
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        logger.info("Starting comprehensive security validation...")
        
        # Initialize components
        await self.initialize()
        
        # Run all validation checks
        validation_functions = [
            self.validate_authentication_implementation,
            self.validate_endpoint_security,
            self.validate_secrets_management,
            self.validate_webhook_security,
            self.validate_input_validation,
            self.validate_cors_configuration,
            self.validate_security_headers,
            self.validate_session_management,
            self.validate_gdpr_compliance,
            self.validate_secrets_in_storage,
            self.validate_dependency_security
        ]
        
        for validation_func in validation_functions:
            try:
                if asyncio.iscoroutinefunction(validation_func):
                    await validation_func()
                else:
                    validation_func()
            except Exception as e:
                logger.error(f"Validation error in {validation_func.__name__}: {e}")
                self.add_issue(SecurityIssue(
                    "HIGH", "Validation", f"Validation error: {validation_func.__name__}",
                    f"Error during validation: {e}"
                ))
        
        # Calculate security score
        total_issues = len(self.issues)
        critical_issues = len([i for i in self.issues if i.severity == "CRITICAL"])
        high_issues = len([i for i in self.issues if i.severity == "HIGH"])
        medium_issues = len([i for i in self.issues if i.severity == "MEDIUM"])
        low_issues = len([i for i in self.issues if i.severity == "LOW"])
        
        # Calculate score (100 - weighted penalty)
        penalty = (critical_issues * 25) + (high_issues * 10) + (medium_issues * 5) + (low_issues * 1)
        security_score = max(0, 100 - penalty)
        
        # Determine security level
        if security_score >= 95:
            security_level = "EXCELLENT"
        elif security_score >= 85:
            security_level = "GOOD" 
        elif security_score >= 70:
            security_level = "ACCEPTABLE"
        elif security_score >= 50:
            security_level = "NEEDS_IMPROVEMENT"
        else:
            security_level = "CRITICAL"
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'security_score': security_score,
            'security_level': security_level,
            'total_issues': total_issues,
            'issues_by_severity': {
                'critical': critical_issues,
                'high': high_issues,
                'medium': medium_issues,
                'low': low_issues
            },
            'issues': [issue.to_dict() for issue in self.issues],
            'recommendations': self._generate_recommendations()
        }
        
        logger.info(
            f"Security validation complete: {security_level} (Score: {security_score}/100)",
            total_issues=total_issues,
            critical=critical_issues,
            high=high_issues
        )
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on found issues."""
        recommendations = []
        
        # Count issues by category
        category_counts = {}
        for issue in self.issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        # Generate category-specific recommendations
        if category_counts.get('Authentication', 0) > 0:
            recommendations.append("Implement comprehensive JWT authentication with proper token validation")
        
        if category_counts.get('Secrets', 0) > 0:
            recommendations.append("Migrate all sensitive credentials to encrypted secrets storage")
        
        if category_counts.get('Webhooks', 0) > 0:
            recommendations.append("Implement webhook signature verification and rate limiting")
        
        if category_counts.get('GDPR', 0) > 0:
            recommendations.append("Complete GDPR compliance implementation for data protection")
        
        if category_counts.get('CORS', 0) > 0:
            recommendations.append("Review and tighten CORS configuration for production")
        
        # General recommendations
        critical_count = len([i for i in self.issues if i.severity == "CRITICAL"])
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical security issues immediately")
        
        if not recommendations:
            recommendations.append("Maintain regular security audits and updates")
        
        return recommendations
    
    def print_summary(self, report: Dict[str, Any]):
        """Print security validation summary."""
        print("\n" + "="*70)
        print("SECURITY VALIDATION REPORT")
        print("="*70)
        
        print(f"Security Score: {report['security_score']}/100 ({report['security_level']})")
        print(f"Total Issues Found: {report['total_issues']}")
        
        if report['total_issues'] > 0:
            print("\nIssues by Severity:")
            for severity, count in report['issues_by_severity'].items():
                if count > 0:
                    icon = {
                        'critical': '❌',
                        'high': '🔴', 
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(severity, 'ℹ️')
                    print(f"  {icon} {severity.upper()}: {count}")
            
            print("\nTop Issues:")
            sorted_issues = sorted(self.issues, key=lambda x: {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x.severity, 0), reverse=True)
            for issue in sorted_issues[:5]:  # Show top 5
                print(f"  [{issue.severity}] {issue.title}")
                if issue.fix_suggestion:
                    print(f"    Fix: {issue.fix_suggestion}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*70)
        
        if report['security_score'] < 85:
            print("⚠️  SECURITY IMPROVEMENT NEEDED")
            print("Critical and high-severity issues should be addressed immediately.")
        else:
            print("✅ SECURITY STATUS ACCEPTABLE")
            print("Continue monitoring and addressing remaining issues.")
        
        print("="*70)


async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive security validation for Telegram ML Bot"
    )
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Attempt to fix found issues automatically"
    )
    parser.add_argument(
        "--report", 
        type=str, 
        help="Generate detailed security report to file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed output during validation"
    )
    
    args = parser.parse_args()
    
    # Create validator and run
    validator = SecurityValidator(
        fix_issues=args.fix,
        verbose=args.verbose
    )
    
    try:
        report = await validator.run_validation()
        
        # Print summary
        validator.print_summary(report)
        
        # Save detailed report if requested
        if args.report:
            report_file = Path(args.report)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Detailed report saved to {report_file}")
        
        # Return appropriate exit code
        critical_issues = report['issues_by_severity']['critical']
        high_issues = report['issues_by_severity']['high']
        
        if critical_issues > 0:
            logger.error(f"Found {critical_issues} critical security issues")
            return 2  # Critical issues
        elif high_issues > 0:
            logger.warning(f"Found {high_issues} high-severity security issues")
            return 1  # High issues
        else:
            logger.info("No critical or high-severity issues found")
            return 0  # Success
    
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        return 3  # Validation error


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
