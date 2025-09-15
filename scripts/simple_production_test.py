#!/usr/bin/env python3
"""
ARIA ELITE Simple Production Verification Script
Uses only built-in modules to verify system functionality
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime
import urllib.request
import urllib.error
import socket

class SimpleProductionVerifier:
    """Simple production verification using built-in modules"""
    
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.backend_url = "http://localhost:8000"
        self.supervisor_token = os.getenv('SUPERVISOR_API_TOKEN', 'dev_token')
        self.test_results = {}
        
    def log_test_result(self, test_name: str, success: bool, details: dict):
        """Log test result"""
        self.test_results[test_name] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED - {details.get('error', 'Unknown error')}")
    
    def check_url_accessibility(self, url: str, timeout: int = 10) -> bool:
        """Check if URL is accessible"""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, socket.timeout, socket.error):
            return False
    
    def test_frontend_accessibility(self):
        """Test frontend accessibility"""
        print("🧪 Testing Frontend Accessibility...")
        
        if self.check_url_accessibility(self.base_url):
            self.log_test_result("Frontend Accessibility", True, {
                'url': self.base_url,
                'status': 'accessible'
            })
            return True
        else:
            self.log_test_result("Frontend Accessibility", False, {
                'url': self.base_url,
                'error': 'Frontend not accessible'
            })
            return False
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        print("🧪 Testing Backend Health...")
        
        health_url = f"{self.backend_url}/health"
        if self.check_url_accessibility(health_url):
            self.log_test_result("Backend Health", True, {
                'url': health_url,
                'status': 'healthy'
            })
            return True
        else:
            self.log_test_result("Backend Health", False, {
                'url': health_url,
                'error': 'Backend not accessible'
            })
            return False
    
    def test_api_endpoints(self):
        """Test basic API endpoints"""
        print("🧪 Testing API Endpoints...")
        
        endpoints = [
            f"{self.base_url}/api/health",
            f"{self.base_url}/api/status",
            f"{self.base_url}/api/trades",
            f"{self.base_url}/api/signals",
        ]
        
        results = {}
        for endpoint in endpoints:
            accessible = self.check_url_accessibility(endpoint)
            results[endpoint] = accessible
            
            if accessible:
                print(f"  ✅ {endpoint}")
            else:
                print(f"  ❌ {endpoint}")
        
        success_count = sum(results.values())
        total_count = len(results)
        
        self.log_test_result("API Endpoints", success_count == total_count, {
            'tested_endpoints': total_count,
            'accessible_endpoints': success_count,
            'results': results
        })
        
        return success_count == total_count
    
    def test_configuration_files(self):
        """Test configuration files exist and are valid"""
        print("🧪 Testing Configuration Files...")
        
        config_files = [
            'configs/project_config.json',
            'configs/execution_config.json',
            'configs/strategy_config.json',
            'prisma/schema.prisma'
        ]
        
        results = {}
        for config_file in config_files:
            try:
                if os.path.exists(config_file):
                    # Try to load JSON files
                    if config_file.endswith('.json'):
                        with open(config_file, 'r') as f:
                            json.load(f)
                        results[config_file] = 'valid'
                        print(f"  ✅ {config_file} - valid")
                    else:
                        results[config_file] = 'exists'
                        print(f"  ✅ {config_file} - exists")
                else:
                    results[config_file] = 'missing'
                    print(f"  ❌ {config_file} - missing")
            except Exception as e:
                results[config_file] = f'error: {str(e)}'
                print(f"  ❌ {config_file} - error: {str(e)}")
        
        valid_files = sum(1 for status in results.values() if status in ['valid', 'exists'])
        total_files = len(results)
        
        self.log_test_result("Configuration Files", valid_files == total_files, {
            'total_files': total_files,
            'valid_files': valid_files,
            'results': results
        })
        
        return valid_files == total_files
    
    def test_database_schema(self):
        """Test database schema"""
        print("🧪 Testing Database Schema...")
        
        schema_file = 'prisma/schema.prisma'
        if os.path.exists(schema_file):
            try:
                with open(schema_file, 'r') as f:
                    content = f.read()
                    
                # Check for key models
                required_models = ['User', 'Trade', 'Signal', 'Symbol']
                missing_models = []
                
                for model in required_models:
                    if f'model {model}' not in content:
                        missing_models.append(model)
                
                if missing_models:
                    self.log_test_result("Database Schema", False, {
                        'missing_models': missing_models,
                        'status': 'incomplete'
                    })
                    return False
                else:
                    self.log_test_result("Database Schema", True, {
                        'required_models': required_models,
                        'status': 'complete'
                    })
                    return True
                    
            except Exception as e:
                self.log_test_result("Database Schema", False, {
                    'error': str(e)
                })
                return False
        else:
            self.log_test_result("Database Schema", False, {
                'error': 'Schema file not found'
            })
            return False
    
    def test_environment_variables(self):
        """Test environment variables"""
        print("🧪 Testing Environment Variables...")
        
        required_vars = [
            'SUPERVISOR_API_TOKEN',
            'GEMINI_API_KEY',
            'MT5_LOGIN',
            'MT5_PASSWORD',
            'MT5_SERVER'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.log_test_result("Environment Variables", False, {
                'missing_variables': missing_vars,
                'status': 'incomplete'
            })
            return False
        else:
            self.log_test_result("Environment Variables", True, {
                'required_variables': required_vars,
                'status': 'complete'
            })
            return False
    
    def test_critical_components(self):
        """Test critical components exist"""
        print("🧪 Testing Critical Components...")
        
        components = [
            'src/app/page.tsx',
            'src/components/TradingChart.tsx',
            'src/components/BackendControlPanel.tsx',
            'src/app/api/trades/route.ts',
            'src/app/api/status/route.ts',
            'backend/app/main.py',
            'backend/core/execution_engine.py',
            'backend/core/mt5_bridge.py',
            'backend/core/gemini_workflow_agent.py'
        ]
        
        missing_components = []
        for component in components:
            if not os.path.exists(component):
                missing_components.append(component)
        
        if missing_components:
            self.log_test_result("Critical Components", False, {
                'missing_components': missing_components,
                'status': 'incomplete'
            })
            return False
        else:
            self.log_test_result("Critical Components", True, {
                'checked_components': components,
                'status': 'complete'
            })
            return False
    
    def test_package_dependencies(self):
        """Test package dependencies"""
        print("🧪 Testing Package Dependencies...")
        
        package_files = ['package.json', 'backend/requirements.txt']
        
        results = {}
        for package_file in package_files:
            if os.path.exists(package_file):
                try:
                    with open(package_file, 'r') as f:
                        content = f.read()
                        # Basic check if it's a valid JSON or requirements file
                        if package_file.endswith('.json'):
                            json.loads(content)
                            results[package_file] = 'valid_json'
                        else:
                            results[package_file] = 'valid_requirements'
                        print(f"  ✅ {package_file}")
                except Exception as e:
                    results[package_file] = f'error: {str(e)}'
                    print(f"  ❌ {package_file} - error: {str(e)}")
            else:
                results[package_file] = 'missing'
                print(f"  ❌ {package_file} - missing")
        
        valid_files = sum(1 for status in results.values() if status.startswith('valid'))
        total_files = len(results)
        
        self.log_test_result("Package Dependencies", valid_files == total_files, {
            'total_files': total_files,
            'valid_files': valid_files,
            'results': results
        })
        
        return valid_files == total_files
    
    def run_verification(self):
        """Run complete verification"""
        print("🚀 Starting ARIA ELITE Simple Production Verification")
        print("=" * 60)
        
        tests = [
            ("Frontend Accessibility", self.test_frontend_accessibility),
            ("Backend Health", self.test_backend_health),
            ("API Endpoints", self.test_api_endpoints),
            ("Configuration Files", self.test_configuration_files),
            ("Database Schema", self.test_database_schema),
            ("Environment Variables", self.test_environment_variables),
            ("Critical Components", self.test_critical_components),
            ("Package Dependencies", self.test_package_dependencies),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} execution failed: {str(e)}")
                results.append((test_name, False))
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 PRODUCTION VERIFICATION SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for _, result in results if result)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        # Generate report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'summary': results
        }
        
        with open('simple_production_verification_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Report saved to: simple_production_verification_report.json")
        
        if success_rate >= 90:
            print("\n🎉 SYSTEM PRODUCTION READY!")
            return True
        elif success_rate >= 70:
            print(f"\n⚠️  SYSTEM MOSTLY READY ({success_rate:.1f}%)")
            return False
        else:
            print(f"\n🚨 SYSTEM NOT PRODUCTION READY ({success_rate:.1f}%)")
            return False

def main():
    """Main execution function"""
    verifier = SimpleProductionVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n✅ Production verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Production verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()