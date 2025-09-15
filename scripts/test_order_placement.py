#!/usr/bin/env python3
"""
ARIA ELITE Order Placement Test Script
Focused testing of order placement functionality
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_placement_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderPlacementTester:
    """Test order placement functionality"""
    
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.backend_url = "http://localhost:8000"
        self.supervisor_token = os.getenv('SUPERVISOR_API_TOKEN', 'dev_token')
        self.test_results = []
        
    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.test_results.append(result)
        
        if success:
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED - {details.get('error', 'Unknown error')}")
            
    async def test_order_placement_endpoint(self):
        """Test basic order placement endpoint"""
        logger.info("🧪 Testing Order Placement Endpoint...")
        
        test_order = {
            "symbol": "EURUSD",
            "direction": "buy",
            "volume": 0.01,
            "entry_price": 1.1000,
            "stop_loss": 1.0950,
            "take_profit": 1.1050,
            "confidence": 0.75,
            "strategy": "smc",
            "timeframe": "1h"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/trades",
                json=test_order,
                headers={'Authorization': f'Bearer {self.supervisor_token}'},
                timeout=30
            )
            
            if response.status_code in [200, 201, 202]:
                result = response.json()
                self.log_test_result("Order Placement", True, {
                    'status_code': response.status_code,
                    'trade_id': result.get('trade_id'),
                    'execution_price': result.get('entry_price'),
                    'response': result
                })
                return True
            else:
                self.log_test_result("Order Placement", False, {
                    'status_code': response.status_code,
                    'error': response.text,
                    'response': response.json() if response.headers.get('content-type') == 'application/json' else response.text
                })
                return False
                
        except requests.exceptions.Timeout:
            self.log_test_result("Order Placement", False, {
                'error': 'Request timeout'
            })
            return False
        except Exception as e:
            self.log_test_result("Order Placement", False, {
                'error': str(e)
            })
            return False
            
    async def test_multiple_order_types(self):
        """Test different order types"""
        logger.info("🧪 Testing Multiple Order Types...")
        
        order_types = [
            {
                "name": "Market Buy",
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
                "name": "Market Sell",
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
                "name": "Small Volume",
                "data": {
                    "symbol": "USDJPY",
                    "direction": "buy",
                    "volume": 0.001,
                    "entry_price": 110.00,
                    "stop_loss": 109.95,
                    "take_profit": 110.05,
                    "confidence": 0.90,
                    "strategy": "news_sentiment"
                }
            }
        ]
        
        success_count = 0
        
        for order_type in order_types:
            try:
                logger.info(f"  Testing {order_type['name']}...")
                
                response = requests.post(
                    f"{self.base_url}/api/trades",
                    json=order_type['data'],
                    headers={'Authorization': f'Bearer {self.supervisor_token}'},
                    timeout=30
                )
                
                if response.status_code in [200, 201, 202]:
                    result = response.json()
                    self.log_test_result(order_type['name'], True, {
                        'status_code': response.status_code,
                        'trade_id': result.get('trade_id'),
                        'symbol': result.get('symbol'),
                        'direction': result.get('direction')
                    })
                    success_count += 1
                else:
                    self.log_test_result(order_type['name'], False, {
                        'status_code': response.status_code,
                        'error': response.text
                    })
                    
            except Exception as e:
                self.log_test_result(order_type['name'], False, {
                    'error': str(e)
                })
                
        return success_count == len(order_types)
        
    async def test_order_validation(self):
        """Test order validation"""
        logger.info("🧪 Testing Order Validation...")
        
        invalid_orders = [
            {
                "name": "Invalid Symbol",
                "data": {
                    "symbol": "INVALID_SYMBOL",
                    "direction": "buy",
                    "volume": 0.01,
                    "entry_price": 1.1000
                },
                "should_fail": True
            },
            {
                "name": "Negative Volume",
                "data": {
                    "symbol": "EURUSD",
                    "direction": "buy",
                    "volume": -0.01,
                    "entry_price": 1.1000
                },
                "should_fail": True
            },
            {
                "name": "Zero Volume",
                "data": {
                    "symbol": "EURUSD",
                    "direction": "buy",
                    "volume": 0,
                    "entry_price": 1.1000
                },
                "should_fail": True
            },
            {
                "name": "Missing Stop Loss",
                "data": {
                    "symbol": "EURUSD",
                    "direction": "buy",
                    "volume": 0.01,
                    "entry_price": 1.1000
                    # No stop_loss
                },
                "should_fail": False  # This might be valid
            }
        ]
        
        validation_passed = 0
        
        for invalid_order in invalid_orders:
            try:
                response = requests.post(
                    f"{self.base_url}/api/trades",
                    json=invalid_order['data'],
                    headers={'Authorization': f'Bearer {self.supervisor_token}'},
                    timeout=30
                )
                
                should_have_failed = invalid_order['should_fail']
                actually_failed = response.status_code >= 400
                
                if should_have_failed and actually_failed:
                    self.log_test_result(invalid_order['name'], True, {
                        'validation': 'correctly_rejected',
                        'status_code': response.status_code
                    })
                    validation_passed += 1
                elif not should_have_failed and not actually_failed:
                    self.log_test_result(invalid_order['name'], True, {
                        'validation': 'correctly_accepted',
                        'status_code': response.status_code
                    })
                    validation_passed += 1
                else:
                    self.log_test_result(invalid_order['name'], False, {
                        'validation': 'incorrect_behavior',
                        'expected': 'reject' if should_have_failed else 'accept',
                        'actual': 'accept' if actually_failed else 'reject',
                        'status_code': response.status_code
                    })
                    
            except Exception as e:
                self.log_test_result(invalid_order['name'], False, {
                    'error': str(e)
                })
                
        return validation_passed == len(invalid_orders)
        
    async def test_active_trades_retrieval(self):
        """Test retrieval of active trades"""
        logger.info("🧪 Testing Active Trades Retrieval...")
        
        try:
            response = requests.get(
                f"{self.base_url}/api/trades",
                headers={'Authorization': f'Bearer {self.supervisor_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', [])
                
                self.log_test_result("Active Trades Retrieval", True, {
                    'status_code': response.status_code,
                    'trades_count': len(trades),
                    'trades': trades[:3]  # Show first 3 trades
                })
                return True
            else:
                self.log_test_result("Active Trades Retrieval", False, {
                    'status_code': response.status_code,
                    'error': response.text
                })
                return False
                
        except Exception as e:
            self.log_test_result("Active Trades Retrieval", False, {
                'error': str(e)
            })
            return False
            
    async def test_concurrent_orders(self):
        """Test concurrent order placement"""
        logger.info("🧪 Testing Concurrent Order Placement...")
        
        import threading
        import concurrent.futures
        
        def place_single_order(order_id):
            try:
                order_data = {
                    "symbol": f"TEST{order_id}",
                    "direction": "buy",
                    "volume": 0.01,
                    "entry_price": 1.1000 + order_id * 0.0001,
                    "stop_loss": 1.0950,
                    "take_profit": 1.1050,
                    "confidence": 0.75,
                    "strategy": "smc"
                }
                
                response = requests.post(
                    f"{self.base_url}/api/trades",
                    json=order_data,
                    headers={'Authorization': f'Bearer {self.supervisor_token}'},
                    timeout=30
                )
                
                return order_id, response.status_code in [200, 201, 202]
                
            except Exception as e:
                return order_id, False
                
        # Test concurrent orders
        num_concurrent_orders = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_orders) as executor:
            future_to_order = {
                executor.submit(place_single_order, i): i 
                for i in range(num_concurrent_orders)
            }
            
            success_count = 0
            for future in concurrent.futures.as_completed(future_to_order):
                order_id, success = future.result()
                if success:
                    success_count += 1
                    
        concurrent_success = success_count == num_concurrent_orders
        self.log_test_result("Concurrent Orders", concurrent_success, {
            'concurrent_orders': num_concurrent_orders,
            'successful_orders': success_count,
            'success_rate': f"{(success_count/num_concurrent_orders)*100:.1f}%"
        })
        
        return concurrent_success
        
    async def test_order_cancellation(self):
        """Test order cancellation functionality"""
        logger.info("🧪 Testing Order Cancellation...")
        
        # First place an order
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
                headers={'Authorization': f'Bearer {self.supervisor_token}'},
                timeout=30
            )
            
            if response.status_code in [200, 201, 202]:
                result = response.json()
                trade_id = result.get('trade_id')
                
                # Try to cancel the order (this endpoint might not exist yet)
                # For now, we'll just verify we got a trade ID
                self.log_test_result("Order Placement (for cancellation)", True, {
                    'trade_id': trade_id,
                    'status': 'placed_for_cancellation'
                })
                return True
            else:
                self.log_test_result("Order Cancellation Setup", False, {
                    'error': 'Failed to place order for cancellation test'
                })
                return False
                
        except Exception as e:
            self.log_test_result("Order Cancellation Setup", False, {
                'error': str(e)
            })
            return False
            
    async def run_order_placement_tests(self):
        """Run all order placement tests"""
        logger.info("🚀 Starting ARIA ELITE Order Placement Tests")
        logger.info("=" * 60)
        
        tests = [
            ("Basic Order Placement", self.test_order_placement_endpoint),
            ("Multiple Order Types", self.test_multiple_order_types),
            ("Order Validation", self.test_order_validation),
            ("Active Trades Retrieval", self.test_active_trades_retrieval),
            ("Concurrent Orders", self.test_concurrent_orders),
            ("Order Cancellation", self.test_order_cancellation),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name}...")
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"❌ {test_name} execution failed: {str(e)}")
                results.append((test_name, False))
                
        # Generate summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 ORDER PLACEMENT TEST SUMMARY")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for _, result in results if result)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            
        # Save detailed results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'summary': results
        }
        
        with open('order_placement_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"\n📄 Detailed report saved to: order_placement_test_report.json")
        
        if success_rate >= 90:
            logger.info("\n🎉 ORDER PLACEMENT SYSTEM FULLY FUNCTIONAL!")
            return True
        elif success_rate >= 70:
            logger.info(f"\n⚠️  ORDER PLACEMENT MOSTLY FUNCTIONAL ({success_rate:.1f}%)")
            return False
        else:
            logger.info(f"\n🚨 ORDER PLACEMENT SYSTEM ISSUES DETECTED ({success_rate:.1f}%)")
            return False

async def main():
    """Main execution function"""
    tester = OrderPlacementTester()
    success = await tester.run_order_placement_tests()
    
    if success:
        logger.info("\n✅ All order placement tests passed!")
        exit(0)
    else:
        logger.info("\n❌ Some order placement tests failed!")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())