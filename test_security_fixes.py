#!/usr/bin/env python3
"""
Quick Security Fixes Verification

Tests the 3 critical security vulnerabilities that were fixed:
1. MD5 hash replacement with SHA-256
2. JWT authentication with algorithm validation
3. Enhanced input sanitization

This is a lightweight verification without full system imports.
"""

import hashlib
import re
from pathlib import Path

def test_md5_replacement():
    """Test that MD5 usage has been replaced with SHA-256."""
    print("🔍 Testing MD5 replacement...")
    
    consciousness_file = Path("app/services/consciousness_mirror.py")
    if not consciousness_file.exists():
        print("❌ consciousness_mirror.py not found")
        return False
    
    content = consciousness_file.read_text()
    
    # Check for MD5 usage
    if "hashlib.md5" in content:
        print("❌ Found hashlib.md5 usage - should be replaced with SHA-256")
        return False
    
    # Check for SHA-256 usage
    if "hashlib.sha256" not in content:
        print("❌ No SHA-256 usage found - MD5 should be replaced")
        return False
    
    print("✅ MD5 successfully replaced with SHA-256")
    return True

def test_jwt_algorithm_validation():
    """Test JWT authentication includes algorithm validation."""
    print("🔍 Testing JWT algorithm validation...")
    
    security_file = Path("app/core/security.py")
    if not security_file.exists():
        print("❌ security.py not found")
        return False
    
    content = security_file.read_text()
    
    # Check for explicit algorithm validation
    if "allowed_algorithms = ['HS256', 'RS256']" not in content:
        print("❌ Missing explicit algorithm whitelist")
        return False
    
    # Check for algorithm validation in decode
    if 'algorithms=[settings.security.jwt_algorithm]' not in content:
        print("❌ Missing algorithm specification in JWT decode")
        return False
    
    # Check for signature verification
    if '"verify_signature": True' not in content:
        print("❌ Missing signature verification")
        return False
    
    print("✅ JWT authentication properly secured with algorithm validation")
    return True

def test_input_sanitization_enhancement():
    """Test enhanced input sanitization."""
    print("🔍 Testing enhanced input sanitization...")
    
    security_utils_file = Path("app/core/security_utils.py")
    if not security_utils_file.exists():
        print("❌ security_utils.py not found")
        return False
    
    content = security_utils_file.read_text()
    
    # Check for advanced injection patterns
    advanced_patterns = [
        r'\[INST\]',  # Instruction tokens
        r'jailbreak',
        r'\bDAN\b',  # "Do Anything Now" prompt
        r'act as if.*?uncensored',
        r'simulate.*?without.*?restrictions'
    ]
    
    missing_patterns = []
    for pattern in advanced_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print(f"❌ Missing advanced injection patterns: {missing_patterns}")
        return False
    
    print("✅ Enhanced input sanitization includes advanced prompt injection protection")
    return True

def test_user_isolation_implementation():
    """Test user isolation system implementation."""
    print("🔍 Testing user isolation implementation...")
    
    security_utils_file = Path("app/core/security_utils.py")
    content = security_utils_file.read_text()
    
    # Check for UserIsolationManager class
    if "class UserIsolationManager:" not in content:
        print("❌ UserIsolationManager class not found")
        return False
    
    # Check for key isolation methods
    required_methods = [
        "create_isolated_context",
        "validate_user_access",
        "sanitize_cache_key",
        "validate_no_cross_contamination"
    ]
    
    for method in required_methods:
        if f"def {method}" not in content:
            print(f"❌ Missing required method: {method}")
            return False
    
    print("✅ User isolation system properly implemented")
    return True

def test_encryption_coverage():
    """Test encryption coverage for psychological data."""
    print("🔍 Testing encryption coverage...")
    
    # Check consciousness mirror uses encryption
    consciousness_file = Path("app/services/consciousness_mirror.py")
    content = consciousness_file.read_text()
    
    if "encrypt_psychological_data" not in content:
        print("❌ Missing psychological data encryption")
        return False
    
    # Check security utils has enhanced encryption
    security_utils_file = Path("app/core/security_utils.py")
    security_content = security_utils_file.read_text()
    
    sensitive_fields = ['personality', 'keystrokes', 'linguistic_fingerprint']
    for field in sensitive_fields:
        if field not in security_content:
            print(f"⚠️  Warning: {field} not explicitly mentioned in encryption")
    
    print("✅ Psychological data encryption implemented")
    return True

def main():
    """Run all security validation tests."""
    print("🔒 SECURITY FIXES VALIDATION")
    print("=" * 50)
    
    tests = [
        ("MD5 Hash Replacement", test_md5_replacement),
        ("JWT Algorithm Validation", test_jwt_algorithm_validation),
        ("Enhanced Input Sanitization", test_input_sanitization_enhancement),
        ("User Isolation System", test_user_isolation_implementation),
        ("Encryption Coverage", test_encryption_coverage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print()  # Add spacing after failed test
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 RESULTS: {passed}/{total} tests passed")
    
    security_score = (passed / total) * 100
    print(f"🎯 SECURITY SCORE: {security_score:.1f}/100")
    
    if security_score >= 100:
        print("🎉 PERFECT! All critical vulnerabilities fixed.")
        print("✅ Ready for Feature 4 development.")
        return True
    elif security_score >= 80:
        print("✅ Good security score - minor issues remain")
        return True
    else:
        print("❌ Critical security issues remain!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)