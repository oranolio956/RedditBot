#!/usr/bin/env python3
"""
Import Validation Test Script
Quick validation of import issues blocking test execution.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports that are currently failing."""
    
    print("🔍 Testing Basic Import Chain...")
    
    # Test 1: Database base class
    try:
        from app.database.base import BaseModel
        print("✅ app.database.base.BaseModel - SUCCESS")
    except Exception as e:
        print(f"❌ app.database.base.BaseModel - FAILED: {e}")
        return False
    
    # Test 2: Models package
    try:
        from app.models.user import User
        print("✅ app.models.user.User - SUCCESS")
    except Exception as e:
        print(f"❌ app.models.user.User - FAILED: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Test 3: Synesthesia model (recently fixed)
    try:
        from app.models.synesthesia import SynestheticProfile
        print("✅ app.models.synesthesia.SynestheticProfile - SUCCESS")
    except Exception as e:
        print(f"❌ app.models.synesthesia.SynestheticProfile - FAILED: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Test 4: Main app (critical for all tests)
    try:
        from app.main import app
        print("✅ app.main.app - SUCCESS")
        print(f"   App type: {type(app)}")
        
        # Test route enumeration
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"   Total routes: {len(routes)}")
        
        # Show sample routes
        sample_routes = routes[:5]
        print("   Sample routes:")
        for route in sample_routes:
            print(f"     - {route}")
            
        return True
        
    except Exception as e:
        print(f"❌ app.main.app - FAILED: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_pytest_collection():
    """Test if pytest can collect tests now."""
    
    print("\n🧪 Testing Pytest Collection...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root
        )
        
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            test_count = 0
            for line in output_lines:
                if '<Function' in line or '<Method' in line:
                    test_count += 1
            
            print(f"✅ Pytest collection - SUCCESS")
            print(f"   Collected tests: {test_count}")
            return True
        else:
            print(f"❌ Pytest collection - FAILED")
            print(f"   Exit code: {result.returncode}")
            print(f"   Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Pytest collection - TIMEOUT (30s)")
        return False
    except Exception as e:
        print(f"❌ Pytest collection - ERROR: {e}")
        return False

def find_remaining_import_issues():
    """Find remaining files with import issues."""
    
    print("\n🔍 Scanning for Remaining Import Issues...")
    
    import subprocess
    
    # Look for remaining base_class imports
    try:
        result = subprocess.run(
            ["grep", "-r", "base_class", "app/"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("⚠️  Found remaining base_class imports:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
        else:
            print("✅ No remaining base_class imports found")
            
    except Exception as e:
        print(f"❌ Grep search failed: {e}")
    
    # Look for circular import patterns
    try:
        result = subprocess.run(
            ["grep", "-r", "from app.models import", "app/models/"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("⚠️  Potential circular imports found:")
            for line in result.stdout.strip().split('\n')[:10]:  # Limit output
                print(f"   {line}")
        else:
            print("✅ No obvious circular imports in models")
            
    except Exception as e:
        print(f"❌ Circular import check failed: {e}")

def main():
    """Run all validation tests."""
    
    print("=" * 60)
    print("📋 API Testing Infrastructure Import Validation")
    print("=" * 60)
    
    # Test basic imports
    imports_work = test_basic_imports()
    
    # Test pytest collection if imports work
    if imports_work:
        pytest_works = test_pytest_collection()
    else:
        pytest_works = False
    
    # Find remaining issues
    find_remaining_import_issues()
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if imports_work and pytest_works:
        print("🎉 SUCCESS: Basic testing infrastructure is now functional!")
        print("   Next steps:")
        print("   1. Run: python3 -m pytest tests/test_api_integration.py -v")
        print("   2. Run security tests: python3 -m pytest tests/security/ -v")
        print("   3. Run full test suite: python3 run_comprehensive_tests.py")
    elif imports_work:
        print("⚠️  PARTIAL SUCCESS: Imports work but pytest has issues")
        print("   Next steps:")
        print("   1. Debug pytest configuration")
        print("   2. Check test dependencies")
        print("   3. Verify test database configuration")
    else:
        print("❌ CRITICAL: Import issues still blocking all testing")
        print("   Next steps:")
        print("   1. Fix remaining base_class imports")
        print("   2. Resolve circular import dependencies")
        print("   3. Re-run this validation script")
    
    print("\n📋 For detailed analysis, see: API_TESTING_INFRASTRUCTURE_AUDIT.md")

if __name__ == "__main__":
    main()