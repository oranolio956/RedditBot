#!/usr/bin/env python3
"""
Simplified Security Validation - Code Analysis Based

Validates security fixes through static code analysis without requiring
heavy dependencies like torch, transformers, etc.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SimplifiedSecurityValidator:
    """Code analysis based security validation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "critical_issues": 0,
            "warnings": 0,
            "findings": []
        }
    
    def log_finding(self, test_name: str, status: str, details: str, severity: str = "info"):
        """Log a security finding."""
        finding = {
            "test": test_name,
            "status": status,
            "details": details,
            "severity": severity
        }
        self.results["findings"].append(finding)
        
        if status == "PASS":
            self.results["tests_passed"] += 1
            print(f"✅ {test_name}: {details}")
        elif status == "FAIL":
            self.results["tests_failed"] += 1
            if severity == "critical":
                self.results["critical_issues"] += 1
            print(f"❌ {test_name}: {details}")
        elif status == "WARN":
            self.results["warnings"] += 1
            print(f"⚠️  {test_name}: {details}")
    
    def validate_md5_replacement(self):
        """Check that MD5 has been replaced with SHA-256."""
        print("\n🔍 1. Validating MD5 Replacement...")
        
        # Check consciousness_mirror.py
        consciousness_file = self.project_root / "app" / "services" / "consciousness_mirror.py"
        if not consciousness_file.exists():
            self.log_finding("MD5 Replacement", "FAIL", 
                           "consciousness_mirror.py not found", "critical")
            return
        
        content = consciousness_file.read_text()
        
        # Check for MD5 usage
        md5_patterns = [
            r'hashlib\.md5',
            r'\.md5\(',
            r'MD5',
            r'md5sum'
        ]
        
        md5_found = False
        for pattern in md5_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                md5_found = True
                break
        
        if md5_found:
            self.log_finding("MD5 Usage", "FAIL", 
                           "MD5 usage still found in consciousness_mirror.py", "critical")
        else:
            self.log_finding("MD5 Usage", "PASS", "No MD5 usage found")
        
        # Check for SHA-256 usage
        sha256_patterns = [
            r'hashlib\.sha256',
            r'SHA256',
            r'sha256'
        ]
        
        sha256_found = False
        for pattern in sha256_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                sha256_found = True
                break
        
        if sha256_found:
            self.log_finding("SHA-256 Usage", "PASS", "SHA-256 implementation found")
        else:
            self.log_finding("SHA-256 Usage", "FAIL", 
                           "No SHA-256 usage found - MD5 replacement incomplete", "critical")
    
    def validate_encryption_implementation(self):
        """Check encryption service implementation."""
        print("\n🔍 2. Validating Encryption Implementation...")
        
        security_utils_file = self.project_root / "app" / "core" / "security_utils.py"
        if not security_utils_file.exists():
            self.log_finding("Encryption Service", "FAIL", 
                           "security_utils.py not found", "critical")
            return
        
        content = security_utils_file.read_text()
        
        # Check for encryption service class
        if "class EncryptionService" in content:
            self.log_finding("EncryptionService Class", "PASS", 
                           "EncryptionService class found")
        else:
            self.log_finding("EncryptionService Class", "FAIL", 
                           "EncryptionService class not found", "critical")
            return
        
        # Check for psychological data encryption methods
        required_methods = [
            "encrypt_psychological_data",
            "decrypt_psychological_data"
        ]
        
        for method in required_methods:
            if method in content:
                self.log_finding(f"Method {method}", "PASS", 
                               f"{method} method found")
            else:
                self.log_finding(f"Method {method}", "FAIL", 
                               f"{method} method not found", "critical")
        
        # Check for Fernet usage (AES encryption)
        if "from cryptography.fernet import Fernet" in content:
            self.log_finding("Fernet Encryption", "PASS", 
                           "Fernet encryption library imported")
        else:
            self.log_finding("Fernet Encryption", "WARN", 
                           "Fernet encryption import not found")
    
    def validate_jwt_security(self):
        """Check JWT implementation security."""
        print("\n🔍 3. Validating JWT Security...")
        
        security_file = self.project_root / "app" / "core" / "security.py"
        if not security_file.exists():
            self.log_finding("JWT Security", "FAIL", 
                           "security.py not found", "critical")
            return
        
        content = security_file.read_text()
        
        # Check for JWT functions
        jwt_functions = [
            "create_access_token",
            "verify_token",
            "validate_jwt_configuration"
        ]
        
        for func in jwt_functions:
            if func in content:
                self.log_finding(f"JWT Function {func}", "PASS", 
                               f"{func} function found")
            else:
                self.log_finding(f"JWT Function {func}", "FAIL", 
                               f"{func} function not found", "critical")
        
        # Check for algorithm specification
        if "algorithms=" in content:
            self.log_finding("JWT Algorithm Validation", "PASS", 
                           "JWT algorithm validation found")
        else:
            self.log_finding("JWT Algorithm Validation", "FAIL", 
                           "JWT algorithm validation not found", "critical")
        
        # Check for signature verification
        if "verify_signature" in content or "decode" in content:
            self.log_finding("JWT Signature Verification", "PASS", 
                           "JWT signature verification implemented")
        else:
            self.log_finding("JWT Signature Verification", "WARN", 
                           "JWT signature verification unclear")
    
    def validate_input_sanitization(self):
        """Check input sanitization implementation."""
        print("\n🔍 4. Validating Input Sanitization...")
        
        security_utils_file = self.project_root / "app" / "core" / "security_utils.py"
        if not security_utils_file.exists():
            self.log_finding("Input Sanitization", "FAIL", 
                           "security_utils.py not found", "critical")
            return
        
        content = security_utils_file.read_text()
        
        # Check for InputSanitizer class
        if "class InputSanitizer" in content:
            self.log_finding("InputSanitizer Class", "PASS", 
                           "InputSanitizer class found")
        else:
            self.log_finding("InputSanitizer Class", "FAIL", 
                           "InputSanitizer class not found", "critical")
            return
        
        # Check for sanitization methods
        sanitization_methods = [
            "sanitize_text_input",
            "sanitize_user_input",
            "validate_input"
        ]
        
        method_found = False
        for method in sanitization_methods:
            if method in content:
                self.log_finding(f"Sanitization Method", "PASS", 
                               f"{method} method found")
                method_found = True
                break
        
        if not method_found:
            self.log_finding("Sanitization Methods", "FAIL", 
                           "No sanitization methods found", "critical")
        
        # Check for prompt injection protection
        injection_patterns = [
            "ignore.*instructions",
            "jailbreak",
            "DAN",
            "system.*override"
        ]
        
        injection_protection = False
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                injection_protection = True
                break
        
        if injection_protection:
            self.log_finding("Prompt Injection Protection", "PASS", 
                           "Prompt injection patterns detected in code")
        else:
            self.log_finding("Prompt Injection Protection", "WARN", 
                           "Prompt injection protection patterns not clearly visible")
    
    def validate_user_isolation(self):
        """Check user isolation implementation."""
        print("\n🔍 5. Validating User Isolation...")
        
        security_utils_file = self.project_root / "app" / "core" / "security_utils.py"
        if not security_utils_file.exists():
            self.log_finding("User Isolation", "FAIL", 
                           "security_utils.py not found", "critical")
            return
        
        content = security_utils_file.read_text()
        
        # Check for UserIsolationManager class
        if "class UserIsolationManager" in content:
            self.log_finding("UserIsolationManager Class", "PASS", 
                           "UserIsolationManager class found")
        else:
            self.log_finding("UserIsolationManager Class", "FAIL", 
                           "UserIsolationManager class not found", "critical")
            return
        
        # Check for isolation methods
        isolation_methods = [
            "create_isolated_context",
            "sanitize_cache_key",
            "validate_no_cross_contamination"
        ]
        
        for method in isolation_methods:
            if method in content:
                self.log_finding(f"Isolation Method {method}", "PASS", 
                               f"{method} method found")
            else:
                self.log_finding(f"Isolation Method {method}", "FAIL", 
                               f"{method} method not found", "critical")
    
    def validate_security_middleware(self):
        """Check security middleware implementation."""
        print("\n🔍 6. Validating Security Middleware...")
        
        middleware_dir = self.project_root / "app" / "middleware"
        if not middleware_dir.exists():
            self.log_finding("Security Middleware", "FAIL", 
                           "middleware directory not found", "critical")
            return
        
        # Check for security middleware files
        security_middleware_files = [
            "security_headers.py",
            "input_validation.py",
            "rate_limiting.py",
            "error_handling.py"
        ]
        
        for middleware_file in security_middleware_files:
            file_path = middleware_dir / middleware_file
            if file_path.exists():
                self.log_finding(f"Middleware {middleware_file}", "PASS", 
                               f"{middleware_file} exists")
            else:
                self.log_finding(f"Middleware {middleware_file}", "WARN", 
                               f"{middleware_file} not found")
    
    def check_environment_security(self):
        """Check environment and configuration security."""
        print("\n🔍 7. Validating Environment Security...")
        
        # Check .env.example file
        env_example = self.project_root / ".env.example"
        if env_example.exists():
            content = env_example.read_text()
            
            # Check for placeholder values (good security practice)
            if "your_secret_key_here" in content.lower() or "change_this" in content.lower():
                self.log_finding("Environment Template", "PASS", 
                               "Environment template uses placeholder values")
            else:
                self.log_finding("Environment Template", "WARN", 
                               "Environment template should use placeholder values")
        else:
            self.log_finding("Environment Template", "WARN", 
                           ".env.example file not found")
        
        # Check that .env is not in git
        gitignore_file = self.project_root / ".gitignore"
        if gitignore_file.exists():
            content = gitignore_file.read_text()
            if ".env" in content:
                self.log_finding("Environment Security", "PASS", 
                               ".env file properly gitignored")
            else:
                self.log_finding("Environment Security", "FAIL", 
                               ".env file not in .gitignore", "critical")
        else:
            self.log_finding("Environment Security", "WARN", 
                           ".gitignore file not found")
    
    def calculate_risk_score(self) -> float:
        """Calculate overall security risk score."""
        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        if total_tests == 0:
            return 10.0  # Maximum risk if no tests run
        
        # Base risk from failed tests
        failure_rate = self.results["tests_failed"] / total_tests
        base_risk = failure_rate * 10.0
        
        # Critical issues add significant risk
        critical_risk = self.results["critical_issues"] * 2.0
        
        # Warnings add moderate risk
        warning_risk = self.results["warnings"] * 0.5
        
        total_risk = base_risk + critical_risk + warning_risk
        return min(total_risk, 10.0)  # Cap at 10.0
    
    def generate_report(self):
        """Generate comprehensive security report."""
        risk_score = self.calculate_risk_score()
        
        print("\n" + "="*60)
        print("🔒 SECURITY VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Tests Passed: {self.results['tests_passed']}")
        print(f"  Tests Failed: {self.results['tests_failed']}")
        print(f"  Critical Issues: {self.results['critical_issues']}")
        print(f"  Warnings: {self.results['warnings']}")
        
        print(f"\n🎯 RISK ASSESSMENT:")
        print(f"  Risk Score: {risk_score:.1f}/10.0")
        
        if risk_score <= 3.0:
            status = "GREEN"
            clearance = "YES"
            print(f"  Security Status: {status} ✅")
            print(f"  Feature 4 Clearance: {clearance} ✅")
        elif risk_score <= 6.0:
            status = "YELLOW"
            clearance = "CONDITIONAL"
            print(f"  Security Status: {status} ⚠️")
            print(f"  Feature 4 Clearance: {clearance} ⚠️")
        else:
            status = "RED"
            clearance = "NO"
            print(f"  Security Status: {status} ❌")
            print(f"  Feature 4 Clearance: {clearance} ❌")
        
        print(f"\n📋 DETAILED FINDINGS:")
        for finding in self.results["findings"]:
            emoji = "✅" if finding["status"] == "PASS" else "❌" if finding["status"] == "FAIL" else "⚠️"
            print(f"  {emoji} {finding['test']}: {finding['details']}")
        
        return {
            "risk_score": risk_score,
            "security_status": status,
            "feature_4_clearance": clearance,
            "critical_issues": self.results["critical_issues"],
            "summary": self.results
        }
    
    def run_all_validations(self):
        """Run all security validations."""
        print("🔒 SIMPLIFIED SECURITY VALIDATION")
        print("Analyzing code for security implementation without runtime dependencies")
        print("="*60)
        
        # Run all validation tests
        self.validate_md5_replacement()
        self.validate_encryption_implementation()
        self.validate_jwt_security()
        self.validate_input_sanitization()
        self.validate_user_isolation()
        self.validate_security_middleware()
        self.check_environment_security()
        
        # Generate final report
        return self.generate_report()


def main():
    """Main validation function."""
    validator = SimplifiedSecurityValidator()
    report = validator.run_all_validations()
    
    # Return exit code based on security status
    if report["security_status"] == "GREEN":
        print("\n🎉 SECURITY VALIDATION: PASSED")
        return 0
    elif report["security_status"] == "YELLOW":
        print("\n⚠️  SECURITY VALIDATION: CONDITIONAL PASS")
        return 1
    else:
        print("\n❌ SECURITY VALIDATION: FAILED")
        return 2


if __name__ == "__main__":
    exit(main())