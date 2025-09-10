#!/usr/bin/env python3
"""
Comprehensive Test Runner for Reddit Bot
Runs all test suites and generates coverage reports to verify 80%+ coverage achievement.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=300  # 5 minute timeout per test suite
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Test execution timed out after 5 minutes"
    except Exception as e:
        return 1, "", str(e)

def install_test_dependencies():
    """Install all required testing dependencies."""
    print("📦 Installing test dependencies...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",  # For parallel test execution
        "pytest-mock>=3.10.0",
        "factory-boy>=3.2.0",
        "httpx>=0.24.0",
        "psutil>=5.9.0",
        "numpy>=1.21.0",
        "coverage>=7.0.0",
        "pytest-html>=3.1.0",  # For HTML test reports
        "pytest-json-report>=1.5.0"
    ]
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        exit_code, stdout, stderr = run_command([sys.executable, "-m", "pip", "install", dep])
        if exit_code != 0:
            print(f"  ❌ Failed to install {dep}: {stderr}")
        else:
            print(f"  ✅ Installed {dep}")

def discover_test_files() -> List[str]:
    """Discover all test files in the tests directory."""
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ Tests directory not found!")
        return []
    
    test_files = list(test_dir.glob("test_*.py"))
    print(f"📁 Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    return [str(f) for f in test_files]

def run_test_suite(test_files: List[str]) -> Dict:
    """Run the complete test suite with coverage."""
    print("\n🧪 Running comprehensive test suite...")
    
    # Prepare pytest command with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=.",  # Coverage for all source code
        "--cov-report=html:htmlcov",  # HTML coverage report
        "--cov-report=json:coverage.json",  # JSON coverage report
        "--cov-report=term-missing",  # Terminal coverage report
        "--cov-fail-under=80",  # Fail if coverage below 80%
        "-v",  # Verbose output
        "--tb=short",  # Shorter tracebacks
        "--json-report",  # JSON test report
        "--json-report-file=test_report.json",
        "--html=test_report.html",  # HTML test report
        "--self-contained-html",
        "-x"  # Stop on first failure for debugging
    ] + test_files
    
    print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "execution_time": execution_time,
        "success": exit_code == 0
    }

def parse_coverage_report() -> Dict:
    """Parse the coverage report JSON to extract metrics."""
    try:
        with open("coverage.json", "r") as f:
            coverage_data = json.load(f)
        
        total_coverage = coverage_data["totals"]["percent_covered"]
        covered_lines = coverage_data["totals"]["covered_lines"]
        missing_lines = coverage_data["totals"]["missing_lines"]
        total_lines = covered_lines + missing_lines
        
        # Get file-by-file breakdown
        file_coverage = {}
        for filename, file_data in coverage_data["files"].items():
            if not filename.startswith("tests/"):  # Only source files
                file_coverage[filename] = {
                    "coverage": file_data["summary"]["percent_covered"],
                    "covered_lines": file_data["summary"]["covered_lines"],
                    "missing_lines": file_data["summary"]["missing_lines"]
                }
        
        return {
            "total_coverage": total_coverage,
            "covered_lines": covered_lines,
            "missing_lines": missing_lines,
            "total_lines": total_lines,
            "file_coverage": file_coverage,
            "meets_target": total_coverage >= 80.0
        }
    
    except FileNotFoundError:
        print("❌ Coverage report not found!")
        return {"total_coverage": 0, "meets_target": False}
    except Exception as e:
        print(f"❌ Error parsing coverage report: {e}")
        return {"total_coverage": 0, "meets_target": False}

def parse_test_report() -> Dict:
    """Parse the test execution report."""
    try:
        with open("test_report.json", "r") as f:
            test_data = json.load(f)
        
        return {
            "total_tests": test_data["summary"]["total"],
            "passed": test_data["summary"].get("passed", 0),
            "failed": test_data["summary"].get("failed", 0),
            "skipped": test_data["summary"].get("skipped", 0),
            "errors": test_data["summary"].get("error", 0),
            "duration": test_data["duration"],
            "success_rate": (test_data["summary"].get("passed", 0) / test_data["summary"]["total"]) * 100 if test_data["summary"]["total"] > 0 else 0
        }
    
    except FileNotFoundError:
        print("❌ Test report not found!")
        return {"total_tests": 0, "success_rate": 0}
    except Exception as e:
        print(f"❌ Error parsing test report: {e}")
        return {"total_tests": 0, "success_rate": 0}

def run_individual_test_suites(test_files: List[str]) -> Dict[str, Dict]:
    """Run each test suite individually to identify specific issues."""
    print("\n🔍 Running individual test suites for detailed analysis...")
    
    results = {}
    
    for test_file in test_files:
        print(f"\n  Running {test_file}...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--no-cov"  # Skip coverage for individual runs
        ]
        
        start_time = time.time()
        exit_code, stdout, stderr = run_command(cmd)
        end_time = time.time()
        
        # Count tests in output
        test_count = stdout.count("PASSED") + stdout.count("FAILED") + stdout.count("SKIPPED")
        passed_count = stdout.count("PASSED")
        failed_count = stdout.count("FAILED")
        
        results[test_file] = {
            "success": exit_code == 0,
            "execution_time": end_time - start_time,
            "test_count": test_count,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": (passed_count / test_count * 100) if test_count > 0 else 0
        }
        
        if exit_code == 0:
            print(f"    ✅ {test_file}: {passed_count}/{test_count} tests passed")
        else:
            print(f"    ❌ {test_file}: {failed_count} tests failed")
    
    return results

def generate_comprehensive_report(coverage_data: Dict, test_data: Dict, individual_results: Dict[str, Dict]):
    """Generate a comprehensive test and coverage report."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST COVERAGE REPORT")
    print("="*80)
    
    # Coverage Summary
    print(f"\n🎯 COVERAGE SUMMARY:")
    print(f"  Total Coverage: {coverage_data['total_coverage']:.2f}%")
    print(f"  Target Achievement: {'✅ ACHIEVED' if coverage_data['meets_target'] else '❌ BELOW TARGET'} (80% target)")
    print(f"  Covered Lines: {coverage_data.get('covered_lines', 0):,}")
    print(f"  Missing Lines: {coverage_data.get('missing_lines', 0):,}")
    print(f"  Total Lines: {coverage_data.get('total_lines', 0):,}")
    
    # Test Execution Summary
    print(f"\n🧪 TEST EXECUTION SUMMARY:")
    print(f"  Total Tests: {test_data['total_tests']}")
    print(f"  Passed: {test_data['passed']} ✅")
    print(f"  Failed: {test_data['failed']} ❌")
    print(f"  Skipped: {test_data['skipped']} ⏭️")
    print(f"  Success Rate: {test_data['success_rate']:.1f}%")
    print(f"  Total Duration: {test_data['duration']:.2f}s")
    
    # Individual Test Suite Results
    print(f"\n📁 INDIVIDUAL TEST SUITE RESULTS:")
    for test_file, results in individual_results.items():
        status = "✅" if results['success'] else "❌"
        print(f"  {status} {test_file}:")
        print(f"    Tests: {results['passed']}/{results['test_count']} passed ({results['success_rate']:.1f}%)")
        print(f"    Duration: {results['execution_time']:.2f}s")
    
    # File-by-File Coverage (top 10 lowest coverage)
    if 'file_coverage' in coverage_data:
        print(f"\n📄 FILES WITH LOWEST COVERAGE (Top 10):")
        sorted_files = sorted(
            coverage_data['file_coverage'].items(),
            key=lambda x: x[1]['coverage']
        )[:10]
        
        for filename, file_data in sorted_files:
            print(f"  {file_data['coverage']:5.1f}% - {filename}")
    
    # Performance Metrics
    total_execution_time = sum(r['execution_time'] for r in individual_results.values())
    avg_test_time = total_execution_time / len(individual_results) if individual_results else 0
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"  Total Test Execution Time: {total_execution_time:.2f}s")
    print(f"  Average Test Suite Time: {avg_test_time:.2f}s")
    print(f"  Tests per Second: {test_data['total_tests'] / total_execution_time:.2f}" if total_execution_time > 0 else "  Tests per Second: N/A")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if coverage_data['total_coverage'] < 80:
        print("  - Increase test coverage to meet 80% target")
        if 'file_coverage' in coverage_data:
            low_coverage_files = [f for f, d in coverage_data['file_coverage'].items() if d['coverage'] < 70]
            if low_coverage_files:
                print(f"  - Focus on files with coverage < 70%: {len(low_coverage_files)} files")
    
    if test_data['failed'] > 0:
        print(f"  - Fix {test_data['failed']} failing tests")
    
    if avg_test_time > 30:
        print("  - Consider optimizing slow test suites (>30s)")
    
    # Overall Status
    print(f"\n🎯 OVERALL STATUS:")
    overall_success = (
        coverage_data['meets_target'] and 
        test_data['failed'] == 0 and 
        test_data['success_rate'] >= 95
    )
    
    if overall_success:
        print("  ✅ ALL TARGETS ACHIEVED!")
        print("  🎉 The Reddit bot has comprehensive test coverage with 80%+ coverage")
    else:
        print("  ❌ Targets not fully met - see recommendations above")
    
    print("="*80)

def main():
    """Main test runner execution."""
    print("🚀 Reddit Bot Comprehensive Test Suite Runner")
    print("="*50)
    
    # Change to project directory
    os.chdir("/Users/daltonmetzler/Desktop/Reddit - bot")
    
    # Install dependencies
    install_test_dependencies()
    
    # Discover test files
    test_files = discover_test_files()
    if not test_files:
        print("❌ No test files found. Exiting.")
        sys.exit(1)
    
    # Run individual test suites first for detailed analysis
    individual_results = run_individual_test_suites(test_files)
    
    # Run comprehensive test suite with coverage
    print("\n🎯 Running comprehensive test suite with coverage analysis...")
    test_result = run_test_suite(test_files)
    
    if not test_result["success"]:
        print(f"❌ Test execution failed:")
        print(f"STDOUT: {test_result['stdout']}")
        print(f"STDERR: {test_result['stderr']}")
    else:
        print("✅ Test execution completed successfully!")
    
    # Parse reports
    coverage_data = parse_coverage_report()
    test_data = parse_test_report()
    
    # Generate comprehensive report
    generate_comprehensive_report(coverage_data, test_data, individual_results)
    
    # Exit with appropriate code
    if coverage_data['meets_target'] and test_data['failed'] == 0:
        print("\n🎉 SUCCESS: All tests passed with 80%+ coverage achieved!")
        sys.exit(0)
    else:
        print("\n❌ INCOMPLETE: Review the report above for areas needing improvement")
        sys.exit(1)

if __name__ == "__main__":
    main()