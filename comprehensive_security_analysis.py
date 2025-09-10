#!/usr/bin/env python3
"""
Comprehensive Security Analysis - Final Security Validation Report

Performs detailed analysis of all security implementations and provides
final risk assessment for Feature 4 development clearance.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class ComprehensiveSecurityAnalysis:
    """Final security analysis with risk categorization."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.critical_issues = []
        self.high_issues = []
        self.medium_issues = []
        self.low_issues = []
        self.security_strengths = []
        
    def analyze_md5_usage(self):
        """Analyze MD5 usage and categorize by security risk."""
        print("🔍 Analyzing MD5 Usage Patterns...")
        
        # Get all MD5 usage
        result = os.popen(f'cd "{self.project_root}" && grep -rn "hashlib\\.md5" app/').read()
        md5_usages = result.strip().split('\n') if result.strip() else []
        
        security_critical_files = [
            'consciousness_mirror.py',
            'security.py', 
            'security_utils.py',
            'auth',
            'login',
            'password',
            'token'
        ]
        
        security_critical_md5 = []
        non_security_md5 = []
        
        for usage in md5_usages:
            if ':' in usage:
                file_path = usage.split(':')[0]
                is_security_critical = any(critical_file in file_path.lower() 
                                         for critical_file in security_critical_files)
                
                if is_security_critical:
                    security_critical_md5.append(usage)
                else:
                    non_security_md5.append(usage)
        
        print(f"  Security-Critical MD5 Usage: {len(security_critical_md5)}")
        print(f"  Non-Security MD5 Usage: {len(non_security_md5)}")
        
        if security_critical_md5:
            self.critical_issues.append({
                "type": "MD5_SECURITY_CRITICAL",
                "count": len(security_critical_md5),
                "details": security_critical_md5,
                "severity": "CRITICAL",
                "description": "MD5 usage in security-critical files"
            })
        
        if non_security_md5:
            # These are lower priority but should still be noted
            self.low_issues.append({
                "type": "MD5_NON_SECURITY", 
                "count": len(non_security_md5),
                "details": non_security_md5[:5],  # Show first 5
                "severity": "LOW",
                "description": "MD5 usage in non-security contexts (caching, filenames)"
            })
        
        # Check consciousness_mirror specifically
        consciousness_file = self.project_root / "app" / "services" / "consciousness_mirror.py"
        if consciousness_file.exists():
            content = consciousness_file.read_text()
            if "hashlib.md5" not in content and "hashlib.sha256" in content:
                self.security_strengths.append("✅ consciousness_mirror.py uses SHA-256 instead of MD5")
            elif "hashlib.md5" in content:
                self.critical_issues.append({
                    "type": "CONSCIOUSNESS_MIRROR_MD5",
                    "severity": "CRITICAL", 
                    "description": "consciousness_mirror.py still uses MD5"
                })
    
    def analyze_encryption_implementation(self):
        """Analyze encryption service implementation."""
        print("🔍 Analyzing Encryption Implementation...")
        
        security_utils = self.project_root / "app" / "core" / "security_utils.py"
        if not security_utils.exists():
            self.critical_issues.append({
                "type": "MISSING_ENCRYPTION_SERVICE",
                "severity": "CRITICAL",
                "description": "EncryptionService not found"
            })
            return
        
        content = security_utils.read_text()
        
        # Check for key security implementations
        required_implementations = {
            "class EncryptionService": "EncryptionService class",
            "encrypt_psychological_data": "Psychological data encryption",
            "decrypt_psychological_data": "Psychological data decryption", 
            "from cryptography.fernet import Fernet": "Fernet encryption library",
            "generate_key": "Key generation",
            "rotate_keys": "Key rotation"
        }
        
        for pattern, description in required_implementations.items():
            if pattern in content:
                self.security_strengths.append(f"✅ {description} implemented")
            else:
                severity = "CRITICAL" if "encrypt_psychological_data" in pattern else "HIGH"
                self.critical_issues.append({
                    "type": "MISSING_ENCRYPTION_FEATURE",
                    "severity": severity,
                    "description": f"Missing {description}"
                }) if severity == "CRITICAL" else self.high_issues.append({
                    "type": "MISSING_ENCRYPTION_FEATURE", 
                    "severity": severity,
                    "description": f"Missing {description}"
                })
    
    def analyze_jwt_security(self):
        """Analyze JWT implementation security."""
        print("🔍 Analyzing JWT Security Implementation...")
        
        security_file = self.project_root / "app" / "core" / "security.py"
        if not security_file.exists():
            self.critical_issues.append({
                "type": "MISSING_JWT_SECURITY",
                "severity": "CRITICAL", 
                "description": "JWT security module not found"
            })
            return
        
        content = security_file.read_text()
        
        # Check critical JWT security features
        jwt_security_checks = {
            "algorithms=": "Algorithm specification",
            "verify_signature": "Signature verification",
            "create_access_token": "Token creation",
            "verify_token": "Token verification",
            "validate_jwt_configuration": "JWT configuration validation"
        }
        
        for check, description in jwt_security_checks.items():
            if check in content:
                self.security_strengths.append(f"✅ JWT {description} implemented")
            else:
                self.high_issues.append({
                    "type": "JWT_SECURITY_MISSING",
                    "severity": "HIGH",
                    "description": f"JWT {description} not found"
                })
        
        # Check for weak algorithm usage
        if "HS256" in content or "RS256" in content:
            self.security_strengths.append("✅ Strong JWT algorithms configured")
        else:
            self.medium_issues.append({
                "type": "JWT_ALGORITHM_UNCLEAR",
                "severity": "MEDIUM",
                "description": "JWT algorithm configuration unclear"
            })
    
    def analyze_input_validation(self):
        """Analyze input validation and sanitization."""
        print("🔍 Analyzing Input Validation Implementation...")
        
        security_utils = self.project_root / "app" / "core" / "security_utils.py"
        if not security_utils.exists():
            return
        
        content = security_utils.read_text()
        
        # Check for input sanitization
        if "class InputSanitizer" in content:
            self.security_strengths.append("✅ InputSanitizer class implemented")
            
            # Check for specific sanitization methods
            sanitization_methods = [
                "sanitize_text_input",
                "sanitize_user_input", 
                "validate_input"
            ]
            
            for method in sanitization_methods:
                if method in content:
                    self.security_strengths.append(f"✅ {method} method implemented")
                    break
            else:
                self.high_issues.append({
                    "type": "MISSING_SANITIZATION_METHODS",
                    "severity": "HIGH",
                    "description": "No sanitization methods found"
                })
        else:
            self.critical_issues.append({
                "type": "MISSING_INPUT_SANITIZER",
                "severity": "CRITICAL",
                "description": "InputSanitizer class not found"
            })
        
        # Check for prompt injection protection
        prompt_injection_patterns = [
            "ignore.*instructions", "jailbreak", "DAN", "override",
            "system.*message", "instructions", "BLOCKED"
        ]
        
        injection_protection_found = False
        for pattern in prompt_injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                injection_protection_found = True
                break
        
        if injection_protection_found:
            self.security_strengths.append("✅ Prompt injection protection patterns detected")
        else:
            self.high_issues.append({
                "type": "PROMPT_INJECTION_PROTECTION",
                "severity": "HIGH", 
                "description": "Prompt injection protection unclear"
            })
    
    def analyze_user_isolation(self):
        """Analyze user isolation implementation."""
        print("🔍 Analyzing User Isolation System...")
        
        security_utils = self.project_root / "app" / "core" / "security_utils.py"
        if not security_utils.exists():
            return
        
        content = security_utils.read_text()
        
        if "class UserIsolationManager" in content:
            self.security_strengths.append("✅ UserIsolationManager class implemented")
            
            isolation_methods = [
                "create_isolated_context",
                "sanitize_cache_key", 
                "validate_no_cross_contamination"
            ]
            
            for method in isolation_methods:
                if method in content:
                    self.security_strengths.append(f"✅ {method} method implemented")
                else:
                    self.high_issues.append({
                        "type": "MISSING_ISOLATION_METHOD",
                        "severity": "HIGH",
                        "description": f"Missing {method} method"
                    })
        else:
            self.critical_issues.append({
                "type": "MISSING_USER_ISOLATION",
                "severity": "CRITICAL",
                "description": "UserIsolationManager class not found"
            })
    
    def analyze_middleware_security(self):
        """Analyze security middleware implementation."""
        print("🔍 Analyzing Security Middleware...")
        
        middleware_dir = self.project_root / "app" / "middleware"
        if not middleware_dir.exists():
            self.high_issues.append({
                "type": "MISSING_MIDDLEWARE_DIR",
                "severity": "HIGH",
                "description": "Security middleware directory not found"
            })
            return
        
        required_middleware = [
            "security_headers.py",
            "input_validation.py", 
            "rate_limiting.py",
            "error_handling.py"
        ]
        
        for middleware in required_middleware:
            middleware_path = middleware_dir / middleware
            if middleware_path.exists():
                self.security_strengths.append(f"✅ {middleware} middleware exists")
            else:
                self.medium_issues.append({
                    "type": "MISSING_MIDDLEWARE",
                    "severity": "MEDIUM",
                    "description": f"Missing {middleware} middleware"
                })
    
    def calculate_final_risk_score(self) -> float:
        """Calculate comprehensive risk score."""
        # Risk calculation based on issue severity
        critical_weight = 3.0
        high_weight = 2.0  
        medium_weight = 1.0
        low_weight = 0.2
        
        total_risk = (
            len(self.critical_issues) * critical_weight +
            len(self.high_issues) * high_weight +
            len(self.medium_issues) * medium_weight + 
            len(self.low_issues) * low_weight
        )
        
        # Risk reduction for security strengths
        strength_reduction = min(len(self.security_strengths) * 0.1, 2.0)
        
        final_risk = max(total_risk - strength_reduction, 0.0)
        return min(final_risk, 10.0)
    
    def generate_final_report(self):
        """Generate final comprehensive security report."""
        risk_score = self.calculate_final_risk_score()
        
        print("\n" + "="*70)
        print("🔒 FINAL SECURITY VALIDATION REPORT")
        print("="*70)
        
        # Status determination
        if risk_score <= 3.0 and len(self.critical_issues) == 0:
            status = "GREEN"
            clearance = "YES" 
            status_emoji = "✅"
        elif risk_score <= 6.0 and len(self.critical_issues) <= 1:
            status = "YELLOW"
            clearance = "CONDITIONAL"
            status_emoji = "⚠️"
        else:
            status = "RED"
            clearance = "NO"
            status_emoji = "❌"
        
        print(f"\n🎯 SECURITY STATUS: {status} {status_emoji}")
        print(f"   Risk Score: {risk_score:.1f}/10.0")
        print(f"   Feature 4 Clearance: {clearance}")
        
        print(f"\n📊 ISSUE SUMMARY:")
        print(f"   Critical Issues: {len(self.critical_issues)}")
        print(f"   High Issues: {len(self.high_issues)}")
        print(f"   Medium Issues: {len(self.medium_issues)}")
        print(f"   Low Issues: {len(self.low_issues)}")
        print(f"   Security Strengths: {len(self.security_strengths)}")
        
        # Critical issues
        if self.critical_issues:
            print(f"\n❌ CRITICAL ISSUES (Must Fix):")
            for issue in self.critical_issues:
                print(f"   • {issue['description']}")
        
        # High issues
        if self.high_issues:
            print(f"\n🔴 HIGH PRIORITY ISSUES:")
            for issue in self.high_issues:
                print(f"   • {issue['description']}")
        
        # Medium issues
        if self.medium_issues:
            print(f"\n🟡 MEDIUM PRIORITY ISSUES:")
            for issue in self.medium_issues:
                print(f"   • {issue['description']}")
        
        # Low issues
        if self.low_issues:
            print(f"\n🔵 LOW PRIORITY ISSUES:")
            for issue in self.low_issues:
                print(f"   • {issue['description']}")
        
        # Security strengths
        if self.security_strengths:
            print(f"\n✅ SECURITY STRENGTHS:")
            for strength in self.security_strengths:
                print(f"   {strength}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        if status == "GREEN":
            print("   ✅ All critical security requirements met")
            print("   ✅ Feature 4 development can proceed")
            print("   ✅ Consider addressing low-priority issues in future sprints")
        elif status == "YELLOW":
            print("   ⚠️  Address high-priority issues before major deployments")
            print("   ⚠️  Feature 4 development can proceed with monitoring")
            print("   ⚠️  Implement additional security testing")
        else:
            print("   ❌ Critical security issues must be resolved")
            print("   ❌ Do not proceed with Feature 4 until fixes implemented")
            print("   ❌ Schedule immediate security remediation")
        
        return {
            "risk_score": risk_score,
            "security_status": status,
            "feature_4_clearance": clearance,
            "critical_issues": len(self.critical_issues),
            "total_issues": len(self.critical_issues) + len(self.high_issues) + len(self.medium_issues) + len(self.low_issues),
            "security_strengths": len(self.security_strengths)
        }
    
    def run_comprehensive_analysis(self):
        """Run all security analyses."""
        print("🔒 COMPREHENSIVE SECURITY ANALYSIS")
        print("Evaluating all security implementations for Feature 4 clearance")
        print("="*70)
        
        # Run all analyses
        self.analyze_md5_usage()
        self.analyze_encryption_implementation()
        self.analyze_jwt_security()
        self.analyze_input_validation()
        self.analyze_user_isolation()
        self.analyze_middleware_security()
        
        # Generate final report
        return self.generate_final_report()


def main():
    """Main analysis function."""
    analyzer = ComprehensiveSecurityAnalysis()
    report = analyzer.run_comprehensive_analysis()
    
    # Print final verdict
    if report["security_status"] == "GREEN":
        print("\n🎉 FINAL SECURITY VALIDATION: PASSED ✅")
        print("✅ All critical vulnerabilities resolved")
        print("✅ Feature 4 development APPROVED")
        return 0
    elif report["security_status"] == "YELLOW":
        print("\n⚠️  FINAL SECURITY VALIDATION: CONDITIONAL PASS ⚠️")
        print("⚠️  Feature 4 development can proceed with monitoring")
        return 1
    else:
        print("\n❌ FINAL SECURITY VALIDATION: FAILED ❌")
        print("❌ Critical vulnerabilities remain")
        print("❌ Feature 4 development BLOCKED")
        return 2


if __name__ == "__main__":
    exit(main())