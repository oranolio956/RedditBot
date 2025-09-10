#!/usr/bin/env python3
"""
Production Security Audit Script

Comprehensive security validation for the Telegram ML Bot application.
Checks for common security vulnerabilities and misconfigurations before deployment.

Usage:
    python security_audit.py
    python security_audit.py --fix-issues
    python security_audit.py --report-only
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class SecurityIssue:
    """Represents a security issue found during audit."""
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: str = ""
    auto_fixable: bool = False
    cve_references: List[str] = field(default_factory=list)


class SecurityAuditor:
    """
    Comprehensive security auditor for the Telegram ML Bot.
    
    Performs various security checks:
    - Secret detection and validation
    - Dependency vulnerability scanning
    - Configuration security analysis
    - Code security patterns
    - Infrastructure security review
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues: List[SecurityIssue] = []
        self.stats = {
            'files_scanned': 0,
            'lines_scanned': 0,
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0,
        }
        
        # Initialize patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for security detection."""
        
        # Secret patterns (high-confidence)
        self.secret_patterns = {
            'openai_api_key': re.compile(r'sk-[a-zA-Z0-9]{48}'),
            'stripe_secret_key': re.compile(r'sk_(test|live)_[a-zA-Z0-9]{24,}'),
            'stripe_webhook_secret': re.compile(r'whsec_[a-zA-Z0-9]{32,}'),
            'anthropic_api_key': re.compile(r'sk-ant-[a-zA-Z0-9\-_]{95}'),
            'telegram_bot_token': re.compile(r'\b\d{8,10}:[a-zA-Z0-9_-]{35}\b'),
            'jwt_secret': re.compile(r'["\']?jwt[_-]?secret[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9+/=]{32,}'),
            'database_url': re.compile(r'postgresql://[^:]+:[^@]+@[^/]+/\w+'),
            'redis_url': re.compile(r'redis://:[^@]+@[^/]+'),
            'private_key': re.compile(r'-----BEGIN (RSA |EC |)PRIVATE KEY-----'),
            'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
            'generic_secret': re.compile(r'["\']?(secret|password|key|token)["\']?\s*[:=]\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
        }
        
        # Insecure code patterns
        self.insecure_patterns = {
            'sql_injection': re.compile(r'execute\s*\(\s*f?["\'].*\{.*\}.*["\']', re.IGNORECASE),
            'command_injection': re.compile(r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', re.IGNORECASE),
            'hardcoded_temp': re.compile(r'["\']?/tmp/[^"\']*["\']?'),
            'weak_crypto': re.compile(r'\b(md5|sha1)\b', re.IGNORECASE),
            'debug_mode': re.compile(r'DEBUG\s*=\s*True', re.IGNORECASE),
            'unsafe_eval': re.compile(r'\beval\s*\(', re.IGNORECASE),
            'unsafe_pickle': re.compile(r'pickle\.loads?\s*\(', re.IGNORECASE),
        }
        
        # Configuration security patterns
        self.config_issues = {
            'weak_cors': re.compile(r'allow_origins\s*=\s*\[\s*["\*\'"]\s*\]', re.IGNORECASE),
            'insecure_cookies': re.compile(r'secure\s*=\s*False', re.IGNORECASE),
            'disabled_csrf': re.compile(r'csrf_protection\s*=\s*False', re.IGNORECASE),
        }
        
        # File extensions to scan
        self.scannable_extensions = {'.py', '.yml', '.yaml', '.json', '.env', '.cfg', '.ini', '.conf'}
    
    def run_audit(self, fix_issues: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive security audit.
        
        Args:
            fix_issues: Whether to automatically fix issues where possible
            
        Returns:
            Audit results dictionary
        """
        print(f"{Colors.BOLD}{Colors.BLUE}🔒 Starting Security Audit{Colors.END}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        # Run all security checks
        self._check_secrets_and_credentials()
        self._check_dependency_vulnerabilities()
        self._check_configuration_security()
        self._check_code_security_patterns()
        self._check_file_permissions()
        self._check_docker_security()
        self._check_infrastructure_security()
        
        # Apply fixes if requested
        if fix_issues:
            self._apply_auto_fixes()
        
        # Generate report
        return self._generate_report()
    
    def _check_secrets_and_credentials(self):
        """Check for exposed secrets and credentials."""
        print(f"{Colors.YELLOW}🔍 Checking for exposed secrets...{Colors.END}")
        
        for file_path in self._get_scannable_files():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    self.stats['files_scanned'] += 1
                    self.stats['lines_scanned'] += len(lines)
                    
                    for line_num, line in enumerate(lines, 1):
                        # Check for secrets
                        for pattern_name, pattern in self.secret_patterns.items():
                            if pattern.search(line):
                                # Check if it's in a safe context (env example, test file, etc.)
                                if self._is_safe_context(file_path, line):
                                    continue
                                
                                self.issues.append(SecurityIssue(
                                    category="secrets",
                                    severity="CRITICAL",
                                    title=f"Exposed {pattern_name.replace('_', ' ').title()}",
                                    description=f"Found potential {pattern_name} in code",
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    recommendation=f"Move {pattern_name} to secure secrets management system",
                                    auto_fixable=False
                                ))
                                self.stats['critical_issues'] += 1
                                
            except Exception as e:
                print(f"{Colors.RED}Error scanning {file_path}: {e}{Colors.END}")
    
    def _check_dependency_vulnerabilities(self):
        """Check for known vulnerabilities in dependencies."""
        print(f"{Colors.YELLOW}🔍 Checking dependency vulnerabilities...{Colors.END}")
        
        requirements_files = [
            self.project_root / "requirements.txt",
            self.project_root / "Pipfile",
            self.project_root / "pyproject.toml"
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    # Use safety to check for vulnerabilities
                    result = subprocess.run(
                        ['safety', 'check', '--file', str(req_file), '--json'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        vulnerabilities = json.loads(result.stdout)
                        for vuln in vulnerabilities:
                            self.issues.append(SecurityIssue(
                                category="dependencies",
                                severity="HIGH",
                                title=f"Vulnerable dependency: {vuln['package_name']}",
                                description=f"Version {vuln['installed_version']} has known vulnerability",
                                recommendation=f"Update to version {vuln['closest_secure_version']} or higher",
                                cve_references=[vuln.get('cve', '')],
                                auto_fixable=False
                            ))
                            self.stats['high_issues'] += 1
                            
                except FileNotFoundError:
                    print(f"{Colors.YELLOW}Warning: 'safety' tool not found. Install with: pip install safety{Colors.END}")
                except Exception as e:
                    print(f"{Colors.YELLOW}Warning: Could not check vulnerabilities: {e}{Colors.END}")
    
    def _check_configuration_security(self):
        """Check configuration files for security issues."""
        print(f"{Colors.YELLOW}🔍 Checking configuration security...{Colors.END}")
        
        config_files = [
            self.project_root / "app" / "config" / "settings.py",
            self.project_root / "app" / "main.py",
            self.project_root / ".env.example",
            self.project_root / "docker-compose.yml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for specific configuration issues
                        if 'DEBUG=true' in content.lower() and 'production' in content.lower():
                            self.issues.append(SecurityIssue(
                                category="configuration",
                                severity="HIGH",
                                title="Debug mode enabled in production config",
                                description="Debug mode should be disabled in production",
                                file_path=str(config_file),
                                recommendation="Set DEBUG=false for production",
                                auto_fixable=True
                            ))
                            self.stats['high_issues'] += 1
                        
                        # Check for weak CORS settings
                        if re.search(r'allow_origins.*\[\s*["\']?\*["\']?\s*\]', content, re.IGNORECASE):
                            self.issues.append(SecurityIssue(
                                category="configuration",
                                severity="MEDIUM",
                                title="Overly permissive CORS settings",
                                description="CORS allows all origins",
                                file_path=str(config_file),
                                recommendation="Restrict CORS to specific domains",
                                auto_fixable=False
                            ))
                            self.stats['medium_issues'] += 1
                            
                except Exception as e:
                    print(f"{Colors.RED}Error checking {config_file}: {e}{Colors.END}")
    
    def _check_code_security_patterns(self):
        """Check for insecure coding patterns."""
        print(f"{Colors.YELLOW}🔍 Checking code security patterns...{Colors.END}")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            # Skip test files and virtual environments
            if any(skip in str(py_file) for skip in ['test_', '__pycache__', 'venv', '.venv', 'node_modules']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        # Check for insecure patterns
                        for pattern_name, pattern in self.insecure_patterns.items():
                            if pattern.search(line):
                                severity = "HIGH" if pattern_name in ['sql_injection', 'command_injection'] else "MEDIUM"
                                
                                self.issues.append(SecurityIssue(
                                    category="code_security",
                                    severity=severity,
                                    title=f"Insecure pattern: {pattern_name.replace('_', ' ').title()}",
                                    description=f"Found potentially insecure {pattern_name} pattern",
                                    file_path=str(py_file),
                                    line_number=line_num,
                                    recommendation=f"Review and secure {pattern_name} usage",
                                    auto_fixable=False
                                ))
                                
                                if severity == "HIGH":
                                    self.stats['high_issues'] += 1
                                else:
                                    self.stats['medium_issues'] += 1
                                    
            except Exception as e:
                print(f"{Colors.RED}Error scanning {py_file}: {e}{Colors.END}")
    
    def _check_file_permissions(self):
        """Check for insecure file permissions."""
        print(f"{Colors.YELLOW}🔍 Checking file permissions...{Colors.END}")
        
        sensitive_files = [
            self.project_root / ".env",
            self.project_root / "app" / "core" / "secrets.py",
        ]
        
        for file_path in sensitive_files:
            if file_path.exists():
                try:
                    stat = file_path.stat()
                    # Check if file is world-readable or world-writable
                    if stat.st_mode & 0o044:  # World or group readable
                        self.issues.append(SecurityIssue(
                            category="file_permissions",
                            severity="HIGH",
                            title="Sensitive file has permissive permissions",
                            description=f"File {file_path.name} is readable by others",
                            file_path=str(file_path),
                            recommendation="Change permissions to 600 (owner read/write only)",
                            auto_fixable=True
                        ))
                        self.stats['high_issues'] += 1
                        
                except Exception as e:
                    print(f"{Colors.RED}Error checking permissions for {file_path}: {e}{Colors.END}")
    
    def _check_docker_security(self):
        """Check Docker configuration security."""
        print(f"{Colors.YELLOW}🔍 Checking Docker security...{Colors.END}")
        
        dockerfile = self.project_root / "Dockerfile"
        docker_compose = self.project_root / "docker-compose.yml"
        
        if dockerfile.exists():
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    
                    # Check for running as root
                    if 'USER root' in content or not re.search(r'USER \w+', content):
                        self.issues.append(SecurityIssue(
                            category="docker_security",
                            severity="MEDIUM",
                            title="Container runs as root",
                            description="Docker container may be running as root user",
                            file_path=str(dockerfile),
                            recommendation="Add non-root user with USER instruction",
                            auto_fixable=False
                        ))
                        self.stats['medium_issues'] += 1
                        
            except Exception as e:
                print(f"{Colors.RED}Error checking Dockerfile: {e}{Colors.END}")
    
    def _check_infrastructure_security(self):
        """Check infrastructure security configuration."""
        print(f"{Colors.YELLOW}🔍 Checking infrastructure security...{Colors.END}")
        
        # Check Kubernetes configurations
        k8s_files = list(self.project_root.glob("k8s/*.yaml")) + list(self.project_root.glob("k8s/*.yml"))
        
        for k8s_file in k8s_files:
            try:
                with open(k8s_file, 'r') as f:
                    content = f.read()
                    
                    # Check for privileged containers
                    if 'privileged: true' in content:
                        self.issues.append(SecurityIssue(
                            category="infrastructure",
                            severity="HIGH",
                            title="Privileged container detected",
                            description="Container running with privileged access",
                            file_path=str(k8s_file),
                            recommendation="Remove privileged access unless absolutely necessary",
                            auto_fixable=False
                        ))
                        self.stats['high_issues'] += 1
                    
                    # Check for missing resource limits
                    if 'kind: Deployment' in content and 'resources:' not in content:
                        self.issues.append(SecurityIssue(
                            category="infrastructure",
                            severity="MEDIUM",
                            title="Missing resource limits",
                            description="Deployment has no CPU/memory limits",
                            file_path=str(k8s_file),
                            recommendation="Add resource limits to prevent resource exhaustion",
                            auto_fixable=False
                        ))
                        self.stats['medium_issues'] += 1
                        
            except Exception as e:
                print(f"{Colors.RED}Error checking {k8s_file}: {e}{Colors.END}")
    
    def _is_safe_context(self, file_path: Path, line: str) -> bool:
        """Check if a potential secret is in a safe context."""
        safe_indicators = [
            '.example' in str(file_path),
            'test' in str(file_path).lower(),
            'template' in str(file_path).lower(),
            'your_' in line.lower(),
            'replace_with' in line.lower(),
            'example' in line.lower(),
            'test_' in line.lower(),
            'sk_test_...' in line,
            'pk_test_...' in line
        ]
        
        return any(safe_indicators)
    
    def _get_scannable_files(self) -> List[Path]:
        """Get list of files to scan for security issues."""
        files = []
        
        for ext in self.scannable_extensions:
            files.extend(self.project_root.rglob(f"*{ext}"))
        
        # Filter out common non-source directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache'}
        
        return [f for f in files if not any(skip_dir in f.parts for skip_dir in skip_dirs)]
    
    def _apply_auto_fixes(self):
        """Apply automatic fixes for fixable issues."""
        print(f"{Colors.GREEN}🔧 Applying automatic fixes...{Colors.END}")
        
        fixes_applied = 0
        
        for issue in self.issues:
            if issue.auto_fixable:
                try:
                    if issue.category == "file_permissions" and issue.file_path:
                        # Fix file permissions
                        os.chmod(issue.file_path, 0o600)
                        print(f"Fixed permissions for {issue.file_path}")
                        fixes_applied += 1
                        
                except Exception as e:
                    print(f"{Colors.RED}Failed to fix {issue.title}: {e}{Colors.END}")
        
        print(f"Applied {fixes_applied} automatic fixes")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        total_issues = len(self.issues)
        
        # Calculate security score (0-100)
        if total_issues == 0:
            security_score = 100
        else:
            # Weight issues by severity
            weighted_score = (
                self.stats['critical_issues'] * 10 +
                self.stats['high_issues'] * 7 +
                self.stats['medium_issues'] * 4 +
                self.stats['low_issues'] * 2
            )
            security_score = max(0, 100 - min(100, weighted_score))
        
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "security_score": security_score,
            "total_issues": total_issues,
            "statistics": self.stats,
            "issues_by_category": self._group_issues_by_category(),
            "issues_by_severity": self._group_issues_by_severity(),
            "detailed_issues": [
                {
                    "category": issue.category,
                    "severity": issue.severity,
                    "title": issue.title,
                    "description": issue.description,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "recommendation": issue.recommendation,
                    "auto_fixable": issue.auto_fixable,
                    "cve_references": issue.cve_references
                }
                for issue in self.issues
            ]
        }
        
        return report
    
    def _group_issues_by_category(self) -> Dict[str, int]:
        """Group issues by category."""
        categories = {}
        for issue in self.issues:
            categories[issue.category] = categories.get(issue.category, 0) + 1
        return categories
    
    def _group_issues_by_severity(self) -> Dict[str, int]:
        """Group issues by severity."""
        severities = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in self.issues:
            severities[issue.severity] = severities.get(issue.severity, 0) + 1
        return severities
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted audit report to console."""
        print("\n" + "=" * 80)
        print(f"{Colors.BOLD}{Colors.BLUE}🔒 SECURITY AUDIT REPORT{Colors.END}")
        print("=" * 80)
        
        # Security score
        score = report['security_score']
        if score >= 90:
            score_color = Colors.GREEN
        elif score >= 70:
            score_color = Colors.YELLOW
        else:
            score_color = Colors.RED
            
        print(f"Security Score: {score_color}{score}/100{Colors.END}")
        print(f"Total Issues: {report['total_issues']}")
        print(f"Files Scanned: {report['statistics']['files_scanned']}")
        print(f"Lines Scanned: {report['statistics']['lines_scanned']:,}")
        
        # Issues by severity
        print(f"\n{Colors.BOLD}Issues by Severity:{Colors.END}")
        severities = report['issues_by_severity']
        print(f"  {Colors.RED}CRITICAL: {severities['CRITICAL']}{Colors.END}")
        print(f"  {Colors.MAGENTA}HIGH:     {severities['HIGH']}{Colors.END}")
        print(f"  {Colors.YELLOW}MEDIUM:   {severities['MEDIUM']}{Colors.END}")
        print(f"  {Colors.CYAN}LOW:      {severities['LOW']}{Colors.END}")
        
        # Issues by category
        print(f"\n{Colors.BOLD}Issues by Category:{Colors.END}")
        for category, count in report['issues_by_category'].items():
            print(f"  {category}: {count}")
        
        # Critical and high severity issues
        critical_and_high = [
            issue for issue in report['detailed_issues'] 
            if issue['severity'] in ['CRITICAL', 'HIGH']
        ]
        
        if critical_and_high:
            print(f"\n{Colors.BOLD}{Colors.RED}🚨 CRITICAL AND HIGH SEVERITY ISSUES:{Colors.END}")
            for issue in critical_and_high[:10]:  # Show first 10
                severity_color = Colors.RED if issue['severity'] == 'CRITICAL' else Colors.MAGENTA
                print(f"\n{severity_color}[{issue['severity']}]{Colors.END} {issue['title']}")
                print(f"  📁 {issue['file_path']}:{issue['line_number'] or 'N/A'}")
                print(f"  📝 {issue['description']}")
                print(f"  💡 {issue['recommendation']}")
        
        # Final recommendations
        print(f"\n{Colors.BOLD}🎯 RECOMMENDATIONS:{Colors.END}")
        
        if report['security_score'] < 70:
            print(f"{Colors.RED}❌ DO NOT DEPLOY TO PRODUCTION{Colors.END}")
            print("   Address critical and high severity issues first")
        elif report['security_score'] < 90:
            print(f"{Colors.YELLOW}⚠️  DEPLOYMENT REQUIRES CAUTION{Colors.END}")
            print("   Review and address security issues")
        else:
            print(f"{Colors.GREEN}✅ SECURITY POSTURE ACCEPTABLE{Colors.END}")
            print("   Continue monitoring for new vulnerabilities")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print("1. Address CRITICAL and HIGH severity issues")
        print("2. Implement proper secrets management")
        print("3. Run regular security scans")
        print("4. Keep dependencies updated")
        print("5. Enable security monitoring in production")


def main():
    """Main entry point for security audit script."""
    parser = argparse.ArgumentParser(description="Security Audit for Telegram ML Bot")
    parser.add_argument("--fix-issues", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--report-only", action="store_true", help="Generate report without console output")
    parser.add_argument("--output", "-o", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = SecurityAuditor()
    report = auditor.run_audit(fix_issues=args.fix_issues)
    
    # Output report
    if not args.report_only:
        auditor.print_report(report)
    
    # Save report to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")
    
    # Exit with error code if critical issues found
    if report['statistics']['critical_issues'] > 0:
        sys.exit(1)
    elif report['statistics']['high_issues'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()