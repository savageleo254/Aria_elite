#!/usr/bin/env python3
"""
ARIA ELITE Multi-AI System Test Script
Comprehensive testing of the new multi-AI browser automation system
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.browser_ai_agent import BrowserAIManager, AIProvider, AccountStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_ai_system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiAITester:
    """Test the multi-AI browser automation system"""
    
    def __init__(self):
        self.browser_ai_manager = BrowserAIManager()
        self.test_results = []
        self.start_time = datetime.now()
        
    async def setup_test_environment(self):
        """Setup test environment with mock accounts"""
        logger.info("🚀 Setting up Multi-AI Test Environment")
        
        try:
            # Initialize the browser AI manager
            await self.browser_ai_manager.initialize()
            logger.info("✅ Browser AI Manager initialized successfully")
            
            # Add some test accounts
            test_accounts = [
                (AIProvider.CHATGPT, "test_chatgpt_1@example.com", "test_password_1", 50),
                (AIProvider.GEMINI, "test_gemini_1@gmail.com", "test_password_1", 100),
                (AIProvider.CLAUDE, "test_claude_1@example.com", "test_password_1", 75),
                (AIProvider.GROK, "test_grok_1@example.com", "test_password_1", 60),
                (AIProvider.PERPLEXITY, "test_perplexity_1@example.com", "test_password_1", 80),
            ]
            
            for provider, email, password, max_usage in test_accounts:
                success = await self.browser_ai_manager.add_account(provider, email, password, max_usage)
                if success:
                    logger.info(f"✅ Added test account for {provider.value}")
                else:
                    logger.warning(f"⚠️ Failed to add test account for {provider.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {str(e)}")
            return False
    
    async def test_system_initialization(self):
        """Test system initialization"""
        logger.info("🧪 Testing System Initialization")
        
        try:
            # Check system status
            status = await self.browser_ai_manager.get_system_status()
            
            if status['browser_ai_manager']['initialized']:
                logger.info("✅ System initialization successful")
                
                # Log provider status
                providers_status = status['browser_ai_manager']['providers_status']
                for provider, info in providers_status.items():
                    logger.info(f"  - {provider}: {info['active']}/{info['accounts']} accounts active")
                
                return True
            else:
                logger.error("❌ System not properly initialized")
                return False
                
        except Exception as e:
            logger.error(f"❌ System initialization test failed: {str(e)}")
            return False
    
    async def test_account_management(self):
        """Test account management functionality"""
        logger.info("🧪 Testing Account Management")
        
        try:
            # Test adding account
            success = await self.browser_ai_manager.add_account(
                AIProvider.CHATGPT, 
                "new_test@example.com", 
                "new_test_password", 
                25
            )
            
            if success:
                logger.info("✅ Account addition successful")
                
                # Test account retrieval
                chatgpt_accounts = self.browser_ai_manager.account_pool.get('chatgpt', [])
                new_account = next((acc for acc in chatgpt_accounts if acc.email == "new_test@example.com"), None)
                
                if new_account:
                    logger.info(f"✅ Account retrieval successful: {new_account.email}")
                    
                    # Test account status
                    if new_account.status == AccountStatus.ACTIVE:
                        logger.info("✅ Account status correct")
                        return True
                    else:
                        logger.error(f"❌ Account status incorrect: {new_account.status}")
                        return False
                else:
                    logger.error("❌ Account retrieval failed")
                    return False
            else:
                logger.error("❌ Account addition failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Account management test failed: {str(e)}")
            return False
    
    async def test_signal_generation(self):
        """Test signal generation with multiple AI providers"""
        logger.info("🧪 Testing Multi-AI Signal Generation")
        
        test_signals = [
            {
                "symbol": "EURUSD",
                "timeframe": "1h",
                "strategy": "smc",
                "parameters": {"confidence": 0.75}
            },
            {
                "symbol": "GBPUSD",
                "timeframe": "4h",
                "strategy": "ai",
                "parameters": {"confidence": 0.80}
            },
            {
                "symbol": "USDJPY",
                "timeframe": "1d",
                "strategy": "news_sentiment",
                "parameters": {"confidence": 0.90}
            }
        ]
        
        successful_signals = 0
        
        for i, signal_request in enumerate(test_signals):
            try:
                logger.info(f"  Testing signal {i+1}/3: {signal_request['symbol']}")
                
                # Generate signal
                signal = await self.browser_ai_manager.generate_signal(
                    symbol=signal_request['symbol'],
                    timeframe=signal_request['timeframe'],
                    strategy=signal_request['strategy'],
                    parameters=signal_request['parameters']
                )
                
                if signal and signal.get('signal_id'):
                    logger.info(f"  ✅ Signal generated successfully: {signal['signal_id']}")
                    logger.info(f"     - Symbol: {signal['symbol']}")
                    logger.info(f"     - Direction: {signal['direction']}")
                    logger.info(f"     - Confidence: {signal['confidence']}")
                    logger.info(f"     - AI Provider: {signal.get('ai_provider', 'unknown')}")
                    
                    successful_signals += 1
                else:
                    logger.error(f"  ❌ Signal generation failed for {signal_request['symbol']}")
                    
            except Exception as e:
                logger.error(f"  ❌ Signal generation error for {signal_request['symbol']}: {str(e)}")
        
        success_rate = (successful_signals / len(test_signals)) * 100
        logger.info(f"Signal Generation Success Rate: {success_rate:.1f}% ({successful_signals}/{len(test_signals)})")
        
        return success_rate >= 66.0  # At least 2 out of 3 should succeed
    
    async def test_provider_rotation(self):
        """Test AI provider rotation logic"""
        logger.info("🧪 Testing Provider Rotation Logic")
        
        try:
            # Test round-robin rotation
            providers_to_test = [AIProvider.CHATGPT, AIProvider.GEMINI, AIProvider.CLAUDE]
            used_accounts = []
            
            for i, provider in enumerate(providers_to_test):
                logger.info(f"  Testing provider {i+1}: {provider.value}")
                
                account = await self.browser_ai_manager.get_best_account(provider)
                
                if account:
                    used_accounts.append(account)
                    logger.info(f"  ✅ Got account for {provider.value}: {account.email}")
                    
                    # Simulate usage
                    account.usage_count += 1
                    account.last_used = datetime.now()
                else:
                    logger.warning(f"  ⚠️ No available account for {provider.value}")
            
            # Check if we got different accounts
            unique_emails = list(set(acc.email for acc in used_accounts))
            logger.info(f"  Used {len(unique_emails)} unique accounts: {unique_emails}")
            
            success = len(used_accounts) >= 2  # At least 2 providers should work
            if success:
                logger.info("✅ Provider rotation test successful")
            else:
                logger.warning("⚠️ Provider rotation test partially successful")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Provider rotation test failed: {str(e)}")
            return False
    
    async def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        logger.info("🧪 Testing Error Handling and Fallbacks")
        
        try:
            # Test with invalid symbol
            invalid_signal = await self.browser_ai_manager.generate_signal(
                symbol="INVALID_SYMBOL",
                timeframe="1h",
                strategy="smc",
                parameters={}
            )
            
            if invalid_signal:
                logger.info("✅ Invalid symbol handled gracefully")
                logger.info(f"   Fallback signal generated: {invalid_signal.get('ai_provider', 'unknown')}")
            else:
                logger.warning("⚠️ No fallback signal generated for invalid symbol")
            
            # Test with missing parameters
            missing_params_signal = await self.browser_ai_manager.generate_signal(
                symbol="EURUSD",
                timeframe="1h",
                strategy="smc",
                parameters=None
            )
            
            if missing_params_signal:
                logger.info("✅ Missing parameters handled gracefully")
            else:
                logger.warning("⚠️ No signal generated for missing parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {str(e)}")
            return False
    
    async def test_performance_tracking(self):
        """Test performance tracking and statistics"""
        logger.info("🧪 Testing Performance Tracking")
        
        try:
            # Get system status to check performance stats
            status = await self.browser_ai_manager.get_system_status()
            
            perf_stats = status['browser_ai_manager']['performance_stats']
            
            logger.info(f"  Total Requests: {perf_stats['total_requests']}")
            logger.info(f"  Successful Requests: {perf_stats['successful_requests']}")
            logger.info(f"  Failed Requests: {perf_stats['failed_requests']}")
            
            # Test provider-specific stats
            provider_stats = perf_stats['provider_stats']
            for provider, stats in provider_stats.items():
                logger.info(f"  {provider}: {stats['requests']} requests, {stats['success']} successful")
            
            # Check if we have any activity
            total_activity = perf_stats['total_requests']
            if total_activity > 0:
                logger.info("✅ Performance tracking active")
                return True
            else:
                logger.warning("⚠️ No performance activity recorded")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance tracking test failed: {str(e)}")
            return False
    
    async def test_concurrent_requests(self):
        """Test concurrent request handling"""
        logger.info("🧪 Testing Concurrent Request Handling")
        
        try:
            import concurrent.futures
            
            def generate_single_signal(symbol):
                return asyncio.create_task(
                    self.browser_ai_manager.generate_signal(
                        symbol=symbol,
                        timeframe="1h",
                        strategy="smc",
                        parameters={}
                    )
                )
            
            # Create multiple concurrent requests
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]
            tasks = [generate_single_signal(symbol) for symbol in symbols]
            
            # Execute concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_results = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"  ❌ Concurrent request {i+1} failed: {str(result)}")
                else:
                    logger.info(f"  ✅ Concurrent request {i+1} successful: {result.get('signal_id', 'N/A')}")
                    successful_results += 1
            
            total_time = end_time - start_time
            logger.info(f"Concurrent processing time: {total_time:.2f} seconds")
            logger.info(f"Success rate: {successful_results}/{len(symbols)} ({(successful_results/len(symbols))*100:.1f}%)")
            
            return successful_results >= 3  # At least 3 should succeed
            
        except Exception as e:
            logger.error(f"❌ Concurrent requests test failed: {str(e)}")
            return False
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("🧹 Cleaning up Test Environment")
        
        try:
            # Close browser instances
            await self.browser_ai_manager.close()
            logger.info("✅ Browser AI Manager closed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive multi-AI system test"""
        logger.info("🚀 Starting ARIA ELITE Multi-AI System Comprehensive Test")
        logger.info("=" * 70)
        
        test_results = {}
        
        # Setup
        if not await self.setup_test_environment():
            logger.error("❌ Test environment setup failed")
            return False
        
        # Run all tests
        tests = [
            ("System Initialization", self.test_system_initialization),
            ("Account Management", self.test_account_management),
            ("Signal Generation", self.test_signal_generation),
            ("Provider Rotation", self.test_provider_rotation),
            ("Error Handling", self.test_error_handling),
            ("Performance Tracking", self.test_performance_tracking),
            ("Concurrent Requests", self.test_concurrent_requests),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name}...")
            try:
                result = await test_func()
                test_results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name}: FAILED - {str(e)}")
                test_results[test_name] = False
        
        # Cleanup
        await self.cleanup_test_environment()
        
        # Generate summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 MULTI-AI SYSTEM TEST SUMMARY")
        logger.info("=" * 70)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        # Generate detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'test_results': test_results,
            'system_info': {
                'project': 'ARIA-ELITE-Multi-AI',
                'version': '2.0.0',
                'test_duration': str(datetime.now() - self.start_time),
                'ai_providers': [provider.value for provider in AIProvider]
            }
        }
        
        with open('multi_ai_system_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved to: multi_ai_system_test_report.json")
        
        if success_rate >= 80:
            logger.info("\n🎉 MULTI-AI SYSTEM MOSTLY READY!")
            logger.info("The system is functional with good performance.")
            return True
        elif success_rate >= 60:
            logger.info(f"\n⚠️  MULTI-AI SYSTEM PARTIALLY READY ({success_rate:.1f}%)")
            logger.info("Some components need attention before production.")
            return False
        else:
            logger.info(f"\n🚨 MULTI-AI SYSTEM NEEDS SIGNIFICANT WORK ({success_rate:.1f}%)")
            logger.info("Multiple components require fixes.")
            return False

async def main():
    """Main execution function"""
    tester = MultiAITester()
    success = await tester.run_comprehensive_test()
    
    if success:
        print("\n✅ Multi-AI system tests completed successfully!")
        exit(0)
    else:
        print("\n❌ Multi-AI system tests failed!")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())