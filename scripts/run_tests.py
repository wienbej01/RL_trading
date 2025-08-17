#!/usr/bin/env python3
"""
Test runner script for RL Intraday Trading System.

This script provides various options for running tests including:
- Unit tests only
- Integration tests only
- Full test suite
- Coverage reporting
- Performance profiling

Usage:
    python scripts/run_tests.py --help
    python scripts/run_tests.py --unit
    python scripts/run_tests.py --integration
    python scripts/run_tests.py --coverage
    python scripts/run_tests.py --profile
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Test runner for the RL trading system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        
    def run_unit_tests(self, verbose: bool = False, coverage: bool = False) -> int:
        """Run unit tests only."""
        cmd = ["python", "-m", "pytest"]
        
        # Add test markers
        cmd.extend(["-m", "unit or not (integration or slow or data or ibkr or live)"])
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
        
        cmd.append(str(self.tests_dir))
        
        logger.info(f"Running unit tests: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests only."""
        cmd = ["python", "-m", "pytest"]
        
        # Add test markers
        cmd.extend(["-m", "integration"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(str(self.tests_dir))
        
        logger.info(f"Running integration tests: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = False) -> int:
        """Run all tests."""
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        
        cmd.append(str(self.tests_dir))
        
        logger.info(f"Running all tests: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_fast_tests(self, verbose: bool = False) -> int:
        """Run fast tests only (exclude slow, data, ibkr, live tests)."""
        cmd = ["python", "-m", "pytest"]
        
        # Exclude slow and external dependency tests
        cmd.extend(["-m", "not (slow or data or ibkr or live)"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(str(self.tests_dir))
        
        logger.info(f"Running fast tests: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """Run a specific test file or test function."""
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(test_path)
        
        logger.info(f"Running specific test: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_with_profile(self, test_type: str = "unit", verbose: bool = False) -> int:
        """Run tests with profiling enabled."""
        cmd = ["python", "-m", "pytest"]
        
        if test_type == "unit":
            cmd.extend(["-m", "unit or not (integration or slow or data or ibkr or live)"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "fast":
            cmd.extend(["-m", "not (slow or data or ibkr or live)"])
        
        if verbose:
            cmd.append("-v")
        
        # Add profiling
        cmd.extend(["--profile", "--profile-svg"])
        
        cmd.append(str(self.tests_dir))
        
        logger.info(f"Running tests with profiling: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def check_test_coverage(self, min_coverage: float = 80.0) -> int:
        """Check test coverage and fail if below threshold."""
        cmd = ["python", "-m", "pytest"]
        
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=xml",
            f"--cov-fail-under={min_coverage}",
            str(self.tests_dir)
        ])
        
        logger.info(f"Checking test coverage (min {min_coverage}%): {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=self.project_root).returncode
    
    def run_linting(self) -> int:
        """Run code linting checks."""
        logger.info("Running code linting checks...")
        
        # Run flake8
        flake8_result = subprocess.run([
            "python", "-m", "flake8", "src", "tests", "scripts"
        ], cwd=self.project_root).returncode
        
        if flake8_result != 0:
            logger.error("Flake8 linting failed")
            return flake8_result
        
        # Run mypy
        mypy_result = subprocess.run([
            "python", "-m", "mypy", "src"
        ], cwd=self.project_root).returncode
        
        if mypy_result != 0:
            logger.error("MyPy type checking failed")
            return mypy_result
        
        logger.info("All linting checks passed")
        return 0
    
    def run_security_checks(self) -> int:
        """Run security vulnerability checks."""
        logger.info("Running security checks...")
        
        # Run bandit
        bandit_result = subprocess.run([
            "python", "-m", "bandit", "-r", "src", "-f", "json", "-o", "bandit-report.json"
        ], cwd=self.project_root).returncode
        
        if bandit_result != 0:
            logger.error("Bandit security check failed")
            return bandit_result
        
        # Run safety
        safety_result = subprocess.run([
            "python", "-m", "safety", "check", "--json", "--output", "safety-report.json"
        ], cwd=self.project_root).returncode
        
        if safety_result != 0:
            logger.warning("Safety check found vulnerabilities (check safety-report.json)")
        
        logger.info("Security checks completed")
        return 0
    
    def generate_test_report(self) -> int:
        """Generate comprehensive test report."""
        logger.info("Generating comprehensive test report...")
        
        cmd = ["python", "-m", "pytest"]
        cmd.extend([
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--junit-xml=test-results.xml",
            "--html=test-report.html",
            "--self-contained-html",
            str(self.tests_dir)
        ])
        
        result = subprocess.run(cmd, cwd=self.project_root).returncode
        
        if result == 0:
            logger.info("Test report generated successfully:")
            logger.info("  - HTML coverage report: htmlcov/index.html")
            logger.info("  - XML coverage report: coverage.xml")
            logger.info("  - JUnit test results: test-results.xml")
            logger.info("  - HTML test report: test-report.html")
        else:
            logger.error("Test report generation failed")
        
        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for RL Intraday Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --unit                 # Run unit tests only
  python scripts/run_tests.py --integration          # Run integration tests only
  python scripts/run_tests.py --fast                 # Run fast tests only
  python scripts/run_tests.py --all --coverage       # Run all tests with coverage
  python scripts/run_tests.py --specific tests/test_config_loader.py
  python scripts/run_tests.py --check-coverage 85   # Require 85% coverage
  python scripts/run_tests.py --lint                 # Run linting checks
  python scripts/run_tests.py --security             # Run security checks
  python scripts/run_tests.py --report               # Generate test report
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests only")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests only")
    test_group.add_argument("--fast", action="store_true", help="Run fast tests only")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--specific", type=str, help="Run specific test file or function")
    
    # Analysis options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--check-coverage", type=float, metavar="PERCENT", 
                       help="Check coverage meets threshold (e.g., 80.0)")
    parser.add_argument("--profile", action="store_true", help="Run with profiling")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    
    # Output options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.lint:
            return runner.run_linting()
        elif args.security:
            return runner.run_security_checks()
        elif args.report:
            return runner.generate_test_report()
        elif args.check_coverage is not None:
            return runner.check_test_coverage(args.check_coverage)
        elif args.specific:
            return runner.run_specific_test(args.specific, args.verbose)
        elif args.unit:
            return runner.run_unit_tests(args.verbose, args.coverage)
        elif args.integration:
            return runner.run_integration_tests(args.verbose)
        elif args.fast:
            return runner.run_fast_tests(args.verbose)
        elif args.all:
            return runner.run_all_tests(args.verbose, args.coverage)
        elif args.profile:
            test_type = "fast" if not (args.unit or args.integration) else ("unit" if args.unit else "integration")
            return runner.run_with_profile(test_type, args.verbose)
        else:
            # Default: run fast tests
            logger.info("No specific test type specified, running fast tests by default")
            return runner.run_fast_tests(args.verbose)
    
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test execution failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())