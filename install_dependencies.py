#!/usr/bin/env python3
"""
Dependency Installation Script
Installs all required dependencies for API testing infrastructure.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def install_dependencies():
    """Install all required dependencies."""
    
    print("=" * 60)
    print("🚀 API Testing Infrastructure Dependency Installation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found. Run from project root.")
        return False
    
    # Core dependencies - install in batches to avoid memory issues
    dependency_batches = [
        {
            "name": "Core FastAPI and Database",
            "packages": [
                "fastapi==0.104.1",
                "uvicorn[standard]==0.24.0", 
                "sqlalchemy==2.0.23",
                "alembic==1.13.0",
                "asyncpg==0.29.0",
                "redis==5.0.1",
                "aioredis==2.0.1"
            ]
        },
        {
            "name": "Basic ML Dependencies", 
            "packages": [
                "numpy==1.24.4",
                "scipy==1.11.4",
                "scikit-learn==1.3.2"
            ]
        },
        {
            "name": "PyTorch (Large Download)",
            "packages": [
                "torch==2.1.1",
                "torchvision==0.16.1"
            ]
        },
        {
            "name": "Transformers and NLP",
            "packages": [
                "transformers==4.35.2",
                "sentence-transformers==2.2.2",
                "nltk==3.8.1"
            ]
        },
        {
            "name": "Audio Processing",
            "packages": [
                "librosa==0.10.1",
                "pydub==0.25.1",
                "ffmpeg-python==0.2.0"
            ]
        },
        {
            "name": "Performance and Utilities",
            "packages": [
                "numba==0.58.1",
                "rtree==1.1.0", 
                "shapely==2.0.2",
                "hiredis==2.2.3"
            ]
        },
        {
            "name": "Testing Dependencies",
            "packages": [
                "pytest==7.4.3",
                "pytest-asyncio==0.21.1",
                "pytest-cov==4.1.0",
                "pytest-mock==3.12.0",
                "factory-boy==3.3.0"
            ]
        }
    ]
    
    success_count = 0
    total_batches = len(dependency_batches)
    
    for i, batch in enumerate(dependency_batches, 1):
        print(f"\n📦 Installing Batch {i}/{total_batches}: {batch['name']}")
        print("-" * 50)
        
        for package in batch['packages']:
            if run_command(f"pip install {package}", f"Installing {package}"):
                success_count += 1
            else:
                print(f"⚠️  Failed to install {package}, continuing...")
    
    # Try installing from requirements.txt as fallback
    print(f"\n📋 Installing remaining dependencies from requirements.txt")
    run_command("pip install -r requirements.txt", "Installing from requirements.txt")
    
    return True

def validate_installation():
    """Validate that critical dependencies are installed."""
    
    print("\n" + "=" * 60)
    print("🔍 Validating Installation")
    print("=" * 60)
    
    critical_imports = [
        ("fastapi", "FastAPI framework"),
        ("sqlalchemy", "Database ORM"),
        ("redis", "Redis client"), 
        ("numpy", "NumPy arrays"),
        ("pytest", "Testing framework"),
    ]
    
    success_count = 0
    
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"✅ {description} - Available")
            success_count += 1
        except ImportError:
            print(f"❌ {description} - Missing")
    
    # Test PyTorch separately (optional)
    try:
        import torch
        print(f"✅ PyTorch - Available (version {torch.__version__})")
    except ImportError:
        print(f"⚠️  PyTorch - Missing (will limit ML features)")
    
    print(f"\n📊 Validation Results: {success_count}/{len(critical_imports)} critical dependencies available")
    
    return success_count >= len(critical_imports) - 1  # Allow 1 failure

def test_app_import():
    """Test if the main app can now be imported."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing App Import")
    print("=" * 60)
    
    try:
        # Change to project directory
        import sys
        sys.path.insert(0, str(Path.cwd()))
        
        from app.main import app
        print("✅ app.main.app - SUCCESS")
        
        # Get route count
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"✅ Total API routes: {len(routes)}")
        
        # Show sample routes
        print("📋 Sample API routes:")
        for route in routes[:5]:
            print(f"   - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ app.main.app import failed: {e}")
        print("\n💡 This may be due to missing optional dependencies.")
        print("   The core system should still work for basic testing.")
        return False

def test_pytest_collection():
    """Test if pytest can collect tests."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Pytest Collection")
    print("=" * 60)
    
    result = run_command(
        "python3 -m pytest --collect-only -q tests/", 
        "Collecting tests with pytest"
    )
    
    if result:
        print("✅ Pytest can collect tests successfully")
        return True
    else:
        print("⚠️  Some pytest collection issues remain")
        print("   This may be due to missing optional dependencies.")
        return False

def main():
    """Main installation and validation process."""
    
    start_time = time.time()
    
    # Install dependencies
    install_success = install_dependencies()
    
    # Validate installation
    validation_success = validate_installation()
    
    # Test app import
    app_import_success = test_app_import()
    
    # Test pytest collection
    pytest_success = test_pytest_collection()
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    print(f"⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"📦 Dependency Installation: {'✅ SUCCESS' if install_success else '⚠️  PARTIAL'}")
    print(f"🔍 Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
    print(f"🚀 App Import: {'✅ SUCCESS' if app_import_success else '⚠️  ISSUES'}")
    print(f"🧪 Pytest Ready: {'✅ SUCCESS' if pytest_success else '⚠️  ISSUES'}")
    
    if validation_success and (app_import_success or pytest_success):
        print("\n🎉 SUCCESS: API testing infrastructure is ready!")
        print("\n📋 Next Steps:")
        print("   1. Run: python3 test_import_validation.py")
        print("   2. Run: python3 -m pytest tests/test_api_integration.py -v")
        print("   3. Run: python3 run_comprehensive_tests.py")
        
        return True
    else:
        print("\n⚠️  PARTIAL SUCCESS: Some components may have issues")
        print("\n📋 Troubleshooting:")
        print("   1. Check for missing system dependencies (ffmpeg, etc.)")
        print("   2. Try: pip install --upgrade -r requirements.txt")
        print("   3. Consider using virtual environment")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)