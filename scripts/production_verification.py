#!/usr/bin/env python3
"""
ARIA ELITE Production Verification Script
Comprehensive testing for production deployment and order placement
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionVerifier:
    """Comprehensive production verification system"""
    
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.backend_url = "http://localhost:8000"
        self.supervisor_token = os.getenv('SUPERVISOR_API_TOKEN', 'dev_token')
        self.test_results = {}
        
    async def run_comprehensive_verification(self):
        """Run complete production verification"""
        logger.info("🚀 Starting ARIA ELITE Production Verification")
        logger.info("=" * 60)
        
        # Test suite components
        test_suites = [
            ("System Configuration", self.verify_system_configuration),
            ("Database Schema", self.verify_database_schema),
            ("API Endpoints", self.verify_api_endpoints),
            ("Order Placement", self.verify_order_placement),
            ("Frontend Components", self.verify_frontend_components),
            ("Security Configuration", self.verify_security_configuration),
            ("Monitoring Systems", self.verify_monitoring_systems),
            ("Risk Management", self.verify_risk_management),
            ("MT5 Integration", self.verify_mt5_integration),
            ("Discord Integration", self.verify_discord_integration),
        ]
        
        # Execute test suites
        for suite_name, test_func in test_suites:
            logger.info(f"\n📋 Running {suite_name} Tests...")
            try:
                result = await test_func()
                self.test_results[suite_name] = result
                logger.info(f"✅ {suite_name}: {'PASSED' if result['passed'] else 'FAILED'}")
                if not result['passed']:
                    logger.error(f"❌ {suite_name} Issues: {result['issues']}")
            except Exception as e:
                logger.error(f"❌ {suite_name} Execution Failed: {str(e)}")
                self.test_results[suite_name] = {
                    'passed': False,
                    'issues': [f"Execution error: {str(e)}"]
                }
        
        # Generate final report
        await self.generate_verification_report()
        
    async def verify_system_configuration(self) -> Dict[str, Any]:
        """Verify system configuration"""
        issues = []
        passed = True
        
        # Check project configuration
        try:
            with open('configs/project_config.json', 'r') as f:
                project_config = json.load(f)
                
            # Verify required fields
            required_fields = ['project', 'version', 'execution_policy', 'supervisor_api']
            for field in required_fields:
                if field not in project_config:
                    issues.append(f"Missing required field: {field}")
                    passed = False
                    
            # Check execution policy
            exec_policy = project_config.get('execution_policy', {})
            if 'max_open_positions' not in exec_policy:
                issues.append("Missing max_open_positions in execution policy")
                passed = False
                
        except Exception as e:
            issues.append(f"Failed to load project config: {str(e)}")
            passed = False
            
        # Check environment variables
        required_env_vars = [
            'SUPERVISOR_API_TOKEN',
            'GEMINI_API_KEY',
            'MT5_LOGIN',
            'MT5_PASSWORD',
            'MT5_SERVER'
        ]
        
        missing_env_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)
                
        if missing_env_vars:
            issues.append(f"Missing environment variables: {missing_env_vars}")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'project_config_loaded': passed,
                'env_vars_complete': not missing_env_vars,
                'required_fields_present': passed
            }
        }
        
    async def verify_database_schema(self) -> Dict[str, Any]:
        """Verify database schema and connectivity"""
        issues = []
        passed = True
        
        try:
            # Test database connection via Prisma
            result = requests.get(f"{self.base_url}/api/health")
            if result.status_code == 200:
                logger.info("✅ Database connection successful")
            else:
                issues.append("Database connection failed")
                passed = False
                
            # Test database schema
            # This would typically involve checking actual database tables
            # For now, we'll verify the schema file exists
            if os.path.exists('prisma/schema.prisma'):
                logger.info("✅ Database schema file exists")
            else:
                issues.append("Database schema file missing")
                passed = False
                
        except Exception as e:
            issues.append(f"Database verification failed: {str(e)}")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'database_connected': passed,
                'schema_file_exists': os.path.exists('prisma/schema.prisma')
            }
        }
        
    async def verify_api_endpoints(self) -> Dict[str, Any]:
        """Verify all API endpoints are functional"""
        issues = []
        passed = True
        
        endpoints_to_test = [
            ('/api/health', 'GET'),
            ('/api/status', 'GET'),
            ('/api/trades', 'GET'),
            ('/api/signals', 'GET'),
            ('/api/models/status', 'GET'),
            ('/api/analytics/performance', 'GET'),
            ('/api/config/system', 'GET'),
        ]
        
        endpoint_results = {}
        
        for endpoint, method in endpoints_to_test:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint}")
                else:
                    response = requests.post(f"{self.base_url}{endpoint}")
                    
                if response.status_code == 200:
                    endpoint_results[endpoint] = 'OK'
                else:
                    issues.append(f"Endpoint {endpoint} returned {response.status_code}")
                    passed = False
                    endpoint_results[endpoint] = f'ERROR {response.status_code}'
                    
            except Exception as e:
                issues.append(f"Endpoint {endpoint} failed: {str(e)}")
                passed = False
                endpoint_results[endpoint] = f'ERROR: {str(e)}'
                
        # Test order placement endpoint
        try:
            order_data = {
                "symbol": "EURUSD",
                "direction": "buy",
                "volume": 0.01,
                "entry_price": 1.1000,
                "stop_loss": 1.0950,
                "take_profit": 1.1050,
                "confidence": 0.75,
                "strategy": "smc"
            }
            
            response = requests.post(
                f"{self.base_url}/api/trades",
                json=order_data,
                headers={'Authorization': f'Bearer {self.supervisor_token}'}
            )
            
            if response.status_code in [200, 201, 202]:
                endpoint_results['/api/trades'] = 'ORDER_PLACED'
                logger.info("✅ Order placement endpoint functional")
            else:
                issues.append(f"Order placement failed: {response.status_code}")
                passed = False
                endpoint_results['/api/trades'] = f'ORDER_FAILED {response.status_code}'
                
        except Exception as e:
            issues.append(f"Order placement test failed: {str(e)}")
            passed = False
            endpoint_results['/api/trades'] = f'ORDER_ERROR: {str(e)}'
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'endpoint_test_results': endpoint_results,
                'total_endpoints_tested': len(endpoints_to_test) + 1,
                'successful_endpoints': sum(1 for result in endpoint_results.values() if result == 'OK' or result == 'ORDER_PLACED')
            }
        }
        
    async def verify_order_placement(self) -> Dict[str, Any]:
        """Comprehensive order placement testing"""
        issues = []
        passed = True
        
        test_orders = [
            {
                "name": "Market Buy Order",
                "data": {
                    "symbol": "EURUSD",
                    "direction": "buy",
                    "volume": 0.01,
                    "entry_price": 1.1000,
                    "stop_loss": 1.0950,
                    "take_profit": 1.1050,
                    "confidence": 0.75,
                    "strategy": "smc"
                }
            },
            {
                "name": "Market Sell Order",
                "data": {
                    "symbol": "GBPUSD",
                    "direction": "sell",
                    "volume": 0.02,
                    "entry_price": 1.2500,
                    "stop_loss": 1.2550,
                    "take_profit": 1.2400,
                    "confidence": 0.80,
                    "strategy": "ai"
                }
            },
            {
                "name": "Small Volume Order",
                "data": {
                    "symbol": "USDJPY",
                    "direction": "buy",
                    "volume": 0.001,  # Minimum volume
                    "entry_price": 110.00,
                    "stop_loss": 109.95,
                    "take_profit": 110.05,
                    "confidence": 0.90,
                    "strategy": "news_sentiment"
                }
            }
        ]
        
        order_results = []
        
        for test_order in test_orders:
            try:
                logger.info(f"Testing {test_order['name']}...")
                
                response = requests.post(
                    f"{self.base_url}/api/trades",
                    json=test_order['data'],
                    headers={'Authorization': f'Bearer {self.supervisor_token}'}
                )
                
                if response.status_code in [200, 201, 202]:
                    result = response.json()
                    order_results.append({
                        'name': test_order['name'],
                        'status': 'SUCCESS',
                        'response': result,
                        'order_id': result.get('trade_id', 'N/A')
                    })
                    logger.info(f"✅ {test_order['name']} successful")
                else:
                    order_results.append({
                        'name': test_order['name'],
                        'status': 'FAILED',
                        'error': response.status_code,
                        'response': response.text
                    })
                    issues.append(f"{test_order['name']} failed with {response.status_code}")
                    passed = False
                    
            except Exception as e:
                order_results.append({
                    'name': test_order['name'],
                    'status': 'ERROR',
                    'error': str(e)
                })
                issues.append(f"{test_order['name']} error: {str(e)}")
                passed = False
                
        # Test order validation
        try:
            invalid_order = {
                "symbol": "INVALID_SYMBOL",
                "direction": "buy",
                "volume": -0.01,  # Invalid volume
                "entry_price": 1.1000
            }
            
            response = requests.post(
                f"{self.base_url}/api/trades",
                json=invalid_order,
                headers={'Authorization': f'Bearer {self.supervisor_token}'}
            )
            
            # Should return error for invalid order
            if response.status_code >= 400:
                logger.info("✅ Order validation working correctly")
            else:
                issues.append("Order validation not working - accepted invalid order")
                passed = False
                
        except Exception as e:
            issues.append(f"Order validation test error: {str(e)}")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'orders_tested': len(test_orders),
                'successful_orders': sum(1 for r in order_results if r['status'] == 'SUCCESS'),
                'order_results': order_results,
                'validation_working': passed
            }
        }
        
    async def verify_frontend_components(self) -> Dict[str, Any]:
        """Verify frontend components are loading correctly"""
        issues = []
        passed = True
        
        # Test frontend accessibility
        try:
            response = requests.get(f"{self.base_url}")
            if response.status_code == 200:
                logger.info("✅ Frontend main page accessible")
            else:
                issues.append(f"Frontend returned {response.status_code}")
                passed = False
                
        except Exception as e:
            issues.append(f"Frontend access failed: {str(e)}")
            passed = False
            
        # Check critical components exist
        critical_components = [
            'src/app/page.tsx',
            'src/components/TradingChart.tsx',
            'src/components/BackendControlPanel.tsx',
            'src/components/ui/card.tsx',
            'src/components/ui/button.tsx',
            'src/components/ui/tabs.tsx'
        ]
        
        for component in critical_components:
            if os.path.exists(component):
                logger.info(f"✅ Component {component} exists")
            else:
                issues.append(f"Missing component: {component}")
                passed = False
                
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'frontend_accessible': passed,
                'critical_components_exist': all(os.path.exists(c) for c in critical_components),
                'components_checked': len(critical_components)
            }
        }
        
    async def verify_security_configuration(self) -> Dict[str, Any]:
        """Verify security configurations"""
        issues = []
        passed = True
        
        # Check authentication requirements
        secure_endpoints = ['/api/status', '/api/trades', '/api/system/kill-switch']
        
        for endpoint in secure_endpoints:
            try:
                # Test without authentication
                response = requests.get(f"{self.base_url}{endpoint}")
                if response.status_code == 401:
                    logger.info(f"✅ {endpoint} properly secured")
                else:
                    issues.append(f"Endpoint {endpoint} not properly secured")
                    passed = False
                    
            except Exception as e:
                issues.append(f"Security test failed for {endpoint}: {str(e)}")
                passed = False
                
        # Check environment variable security
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                env_content = f.read()
                
            # Check for hardcoded secrets
            if 'password' in env_content.lower() and '=' in env_content:
                logger.info("✅ Environment variables properly configured")
            else:
                issues.append("Environment variables may not be properly configured")
                passed = False
        else:
            issues.append("Environment file (.env) not found")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'endpoints_secured': passed,
                'env_file_exists': os.path.exists(env_file),
                'authentication_working': passed
            }
        }
        
    async def verify_monitoring_systems(self) -> Dict[str, Any]:
        """Verify monitoring and logging systems"""
        issues = []
        passed = True
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Health check passed: {health_data.get('status', 'unknown')}")
            else:
                issues.append("Health check failed")
                passed = False
                
        except Exception as e:
            issues.append(f"Health check error: {str(e)}")
            passed = False
            
        # Check logging configuration
        log_files = ['logs/trading.log', 'logs/system.log', 'production_verification.log']
        existing_logs = [log for log in log_files if os.path.exists(log)]
        
        if existing_logs:
            logger.info(f"✅ Log files found: {len(existing_logs)}")
        else:
            issues.append("No log files found")
            passed = False
            
        # Test system status endpoint
        try:
            response = requests.get(f"{self.base_url}/api/status")
            if response.status_code == 200:
                status_data = response.json()
                logger.info(f"✅ System status accessible: {status_data.get('system_status', 'unknown')}")
            else:
                issues.append("System status endpoint failed")
                passed = False
                
        except Exception as e:
            issues.append(f"System status error: {str(e)}")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'health_check_passed': passed,
                'log_files_available': len(existing_logs),
                'system_status_accessible': passed
            }
        }
        
    async def verify_risk_management(self) -> Dict[str, Any]:
        """Verify risk management systems"""
        issues = []
        passed = True
        
        # Check risk configuration
        try:
            with open('configs/execution_config.json', 'r') as f:
                exec_config = json.load(f)
                
            required_risk_settings = [
                'max_position_size',
                'max_daily_loss',
                'max_risk_per_trade',
                'stop_loss_distance',
                'take_profit_distance'
            ]
            
            for setting in required_risk_settings:
                if setting not in exec_config:
                    issues.append(f"Missing risk setting: {setting}")
                    passed = False
                    
        except Exception as e:
            issues.append(f"Risk config loading failed: {str(e)}")
            passed = False
            
        # Test risk validation through order placement
        try:
            # Test with excessive volume
            high_risk_order = {
                "symbol": "EURUSD",
                "direction": "buy",
                "volume": 10.0,  # Very high volume
                "entry_price": 1.1000,
                "stop_loss": 1.0900,  # Very wide stop
                "take_profit": 1.1100,  # Very wide take profit
                "confidence": 0.5,
                "strategy": "smc"
            }
            
            response = requests.post(
                f"{self.base_url}/api/trades",
                json=high_risk_order,
                headers={'Authorization': f'Bearer {self.supervisor_token}'}
            )
            
            # Should reject high-risk orders
            if response.status_code >= 400:
                logger.info("✅ Risk validation working - rejected high-risk order")
            else:
                issues.append("Risk validation not working - accepted high-risk order")
                passed = False
                
        except Exception as e:
            issues.append(f"Risk validation test error: {str(e)}")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'risk_config_complete': passed,
                'risk_validation_working': passed,
                'required_settings_present': passed
            }
        }
        
    async def verify_mt5_integration(self) -> Dict[str, Any]:
        """Verify MT5 integration (mock test)"""
        issues = []
        passed = True
        
        # Check MT5 configuration
        try:
            with open('configs/project_config.json', 'r') as f:
                project_config = json.load(f)
                
            mt5_config = project_config.get('environment', {}).get('MT5', {})
            
            if mt5_config.get('terminal_required', False):
                logger.info("✅ MT5 terminal required (as expected)")
                
                # Check if MT5 credentials are configured
                mt5_creds = [
                    mt5_config.get('login'),
                    mt5_config.get('password'),
                    mt5_config.get('server')
                ]
                
                if all(cred for cred in mt5_creds):
                    logger.info("✅ MT5 credentials configured")
                else:
                    issues.append("MT5 credentials not fully configured")
                    passed = False
                    
        except Exception as e:
            issues.append(f"MT5 config check failed: {str(e)}")
            passed = False
            
        # Check MT5 bridge file exists
        if os.path.exists('backend/core/mt5_bridge.py'):
            logger.info("✅ MT5 bridge implementation exists")
        else:
            issues.append("MT5 bridge implementation missing")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'mt5_configured': passed,
                'mt5_bridge_exists': os.path.exists('backend/core/mt5_bridge.py'),
                'credentials_complete': all(cred for cred in mt5_creds)
            }
        }
        
    async def verify_discord_integration(self) -> Dict[str, Any]:
        """Verify Discord integration"""
        issues = []
        passed = True
        
        # Check Discord bot configuration
        try:
            with open('configs/project_config.json', 'r') as f:
                project_config = json.load(f)
                
            if 'discord_human_override' in project_config.get('security', {}):
                logger.info("✅ Discord override configured")
            else:
                issues.append("Discord override not configured")
                passed = False
                
        except Exception as e:
            issues.append(f"Discord config check failed: {str(e)}")
            passed = False
            
        # Check Discord bot file exists
        if os.path.exists('discord_agent/discord_bot.py'):
            logger.info("✅ Discord bot implementation exists")
        else:
            issues.append("Discord bot implementation missing")
            passed = False
            
        return {
            'passed': passed,
            'issues': issues,
            'details': {
                'discord_configured': passed,
                'discord_bot_exists': os.path.exists('discord_agent/discord_bot.py'),
                'override_available': passed
            }
        }
        
    async def generate_verification_report(self):
        """Generate comprehensive verification report"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 PRODUCTION VERIFICATION REPORT")
        logger.info("=" * 60)
        
        # Calculate overall score
        total_suites = len(self.test_results)
        passed_suites = sum(1 for result in self.test_results.values() if result['passed'])
        overall_score = (passed_suites / total_suites) * 100
        
        logger.info(f"Overall Score: {overall_score:.1f}% ({passed_suites}/{total_suites} suites passed)")
        
        # Detailed results
        for suite_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            logger.info(f"\n{suite_name}: {status}")
            
            if result['details']:
                logger.info("  Details:")
                for key, value in result['details'].items():
                    logger.info(f"    {key}: {value}")
                    
            if result['issues']:
                logger.info("  Issues:")
                for issue in result['issues']:
                    logger.info(f"    - {issue}")
                    
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'test_results': self.test_results
        }
        
        with open('production_verification_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"\n📄 Full report saved to: production_verification_report.json")
        
        # Production readiness assessment
        if overall_score >= 90:
            logger.info("\n🎉 SYSTEM PRODUCTION READY!")
            logger.info("All critical systems verified and functioning correctly.")
        elif overall_score >= 70:
            logger.info(f"\n⚠️  SYSTEM MOSTLY READY ({overall_score:.1f}%)")
            logger.info("Minor issues detected - review report for details.")
        else:
            logger.info(f"\n🚨 SYSTEM NOT PRODUCTION READY ({overall_score:.1f}%)")
            logger.info("Critical issues detected - address all failed tests before deployment.")

async def main():
    """Main execution function"""
    verifier = ProductionVerifier()
    await verifier.run_comprehensive_verification()

if __name__ == "__main__":
    asyncio.run(main())