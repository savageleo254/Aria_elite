import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import sqlite3

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class TestStatus(Enum):
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class TestResult(Enum):
    CHAMPION_WINS = "champion_wins"
    CHALLENGER_WINS = "challenger_wins"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class ABTest:
    test_id: str
    name: str
    champion_strategy: str
    challenger_strategy: str
    start_time: datetime
    end_time: Optional[datetime]
    status: TestStatus
    allocation_ratio: float  # % allocated to challenger (0.1 = 10%)
    min_sample_size: int
    confidence_level: float
    champion_performance: Dict[str, float]
    challenger_performance: Dict[str, float]
    statistical_significance: Optional[float]
    test_result: Optional[TestResult]

@dataclass
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_trade_duration: float
    timestamp: datetime

class ABTestingAgent:
    """
    A/B Strategy Testing Framework for ARIA-DAN
    Champion/Challenger testing with statistical validation
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.active_tests = {}
        self.completed_tests = []
        self.strategy_performance = {}
        
        # Default testing parameters
        self.default_params = {
            'min_sample_size': 100,
            'confidence_level': 0.95,
            'max_test_duration_days': 30,
            'challenger_allocation': 0.2,  # 20% to challenger
            'significance_threshold': 0.05
        }
        
    async def initialize(self):
        """Initialize the A/B testing agent"""
        try:
            logger.info("Initializing A/B Strategy Testing Agent")
            
            await self._load_testing_config()
            await self._initialize_testing_database()
            await self._load_active_tests()
            
            # Start monitoring loop
            self.monitoring_task = asyncio.create_task(self._testing_monitoring_loop())
            
            self.is_initialized = True
            logger.info("A/B Testing Agent initialized - Ready for strategy testing")
            
        except Exception as e:
            logger.error(f"Failed to initialize A/B Testing Agent: {str(e)}")
            raise
    
    async def _load_testing_config(self):
        """Load A/B testing configuration"""
        try:
            testing_config = self.config.get('ab_testing', {})
            
            if 'default_params' in testing_config:
                self.default_params.update(testing_config['default_params'])
            
            logger.info("A/B testing configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading testing config: {str(e)}")
    
    async def _initialize_testing_database(self):
        """Initialize database for A/B testing"""
        try:
            db_path = "db/ab_testing.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    champion_strategy TEXT,
                    challenger_strategy TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    status TEXT,
                    allocation_ratio REAL,
                    min_sample_size INTEGER,
                    confidence_level REAL,
                    test_result TEXT,
                    statistical_significance REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    strategy_name TEXT,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_return REAL,
                    avg_trade_duration REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("A/B testing database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing testing database: {str(e)}")
    
    async def _load_active_tests(self):
        """Load active tests from database"""
        try:
            db_path = "db/ab_testing.db"
            conn = sqlite3.connect(db_path)
            
            query = """
                SELECT * FROM ab_tests 
                WHERE status IN ('setup', 'running', 'paused')
            """
            
            df = pd.read_sql_query(query, conn)
            
            for _, row in df.iterrows():
                test = ABTest(
                    test_id=row['test_id'],
                    name=row['name'],
                    champion_strategy=row['champion_strategy'],
                    challenger_strategy=row['challenger_strategy'],
                    start_time=pd.to_datetime(row['start_time']),
                    end_time=pd.to_datetime(row['end_time']) if row['end_time'] else None,
                    status=TestStatus(row['status']),
                    allocation_ratio=row['allocation_ratio'],
                    min_sample_size=row['min_sample_size'],
                    confidence_level=row['confidence_level'],
                    champion_performance={},
                    challenger_performance={},
                    statistical_significance=row['statistical_significance'],
                    test_result=TestResult(row['test_result']) if row['test_result'] else None
                )
                
                self.active_tests[test.test_id] = test
                
                # Load performance data
                await self._load_test_performance(test.test_id)
            
            conn.close()
            logger.info(f"Loaded {len(self.active_tests)} active tests")
            
        except Exception as e:
            logger.error(f"Error loading active tests: {str(e)}")
    
    async def _load_test_performance(self, test_id: str):
        """Load performance data for specific test"""
        try:
            db_path = "db/ab_testing.db"
            conn = sqlite3.connect(db_path)
            
            query = """
                SELECT * FROM test_performance 
                WHERE test_id = ? 
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(test_id,))
            
            if len(df) > 0:
                test = self.active_tests[test_id]
                
                # Get latest performance for each strategy
                champion_data = df[df['strategy_name'] == test.champion_strategy].iloc[0] if len(df[df['strategy_name'] == test.champion_strategy]) > 0 else None
                challenger_data = df[df['strategy_name'] == test.challenger_strategy].iloc[0] if len(df[df['strategy_name'] == test.challenger_strategy]) > 0 else None
                
                if champion_data is not None:
                    test.champion_performance = {
                        'total_trades': champion_data['total_trades'],
                        'win_rate': champion_data['win_rate'],
                        'profit_factor': champion_data['profit_factor'],
                        'sharpe_ratio': champion_data['sharpe_ratio'],
                        'max_drawdown': champion_data['max_drawdown'],
                        'total_return': champion_data['total_return']
                    }
                
                if challenger_data is not None:
                    test.challenger_performance = {
                        'total_trades': challenger_data['total_trades'],
                        'win_rate': challenger_data['win_rate'],
                        'profit_factor': challenger_data['profit_factor'],
                        'sharpe_ratio': challenger_data['sharpe_ratio'],
                        'max_drawdown': challenger_data['max_drawdown'],
                        'total_return': challenger_data['total_return']
                    }
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading test performance for {test_id}: {str(e)}")
    
    async def _testing_monitoring_loop(self):
        """Main A/B testing monitoring loop"""
        while self.is_initialized:
            try:
                await self._update_test_performance()
                await self._check_test_completion()
                await self._analyze_test_results()
                
                await asyncio.sleep(3600)  # 1-hour cycles
                
            except Exception as e:
                logger.error(f"Error in testing monitoring loop: {str(e)}")
                await asyncio.sleep(300)
    
    async def _update_test_performance(self):
        """Update performance metrics for all active tests"""
        try:
            for test_id, test in self.active_tests.items():
                if test.status == TestStatus.RUNNING:
                    # Simulate performance updates
                    champion_perf = await self._get_strategy_performance(test.champion_strategy)
                    challenger_perf = await self._get_strategy_performance(test.challenger_strategy)
                    
                    if champion_perf:
                        test.champion_performance = champion_perf
                        await self._log_performance(test_id, test.champion_strategy, champion_perf)
                    
                    if challenger_perf:
                        test.challenger_performance = challenger_perf
                        await self._log_performance(test_id, test.challenger_strategy, challenger_perf)
            
        except Exception as e:
            logger.error(f"Error updating test performance: {str(e)}")
    
    async def _get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, float]]:
        """Get current performance metrics for strategy (simulated)"""
        try:
            # Simulate strategy performance with some randomness
            base_performance = {
                'smc_strategy': {'win_rate': 0.65, 'profit_factor': 1.8, 'sharpe_ratio': 1.4},
                'trend_following': {'win_rate': 0.58, 'profit_factor': 2.1, 'sharpe_ratio': 1.2},
                'mean_reversion': {'win_rate': 0.72, 'profit_factor': 1.6, 'sharpe_ratio': 1.1}
            }
            
            if strategy_name not in base_performance:
                return None
            
            base = base_performance[strategy_name]
            
            # Add some variation
            performance = {
                'total_trades': np.random.randint(50, 200),
                'win_rate': base['win_rate'] + np.random.normal(0, 0.05),
                'profit_factor': base['profit_factor'] + np.random.normal(0, 0.2),
                'sharpe_ratio': base['sharpe_ratio'] + np.random.normal(0, 0.1),
                'max_drawdown': np.random.uniform(0.05, 0.15),
                'total_return': np.random.uniform(0.10, 0.30),
                'avg_trade_duration': np.random.uniform(2.0, 8.0)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {str(e)}")
            return None
    
    async def _log_performance(self, test_id: str, strategy_name: str, performance: Dict[str, float]):
        """Log performance data to database"""
        try:
            db_path = "db/ab_testing.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO test_performance 
                (test_id, strategy_name, total_trades, win_rate, profit_factor,
                 sharpe_ratio, max_drawdown, total_return, avg_trade_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id, strategy_name, performance['total_trades'],
                performance['win_rate'], performance['profit_factor'],
                performance['sharpe_ratio'], performance['max_drawdown'],
                performance['total_return'], performance['avg_trade_duration']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging performance: {str(e)}")
    
    async def _check_test_completion(self):
        """Check if any tests should be completed"""
        try:
            current_time = datetime.now()
            
            for test_id, test in list(self.active_tests.items()):
                if test.status != TestStatus.RUNNING:
                    continue
                
                # Check completion conditions
                should_complete = False
                completion_reason = ""
                
                # Check minimum sample size
                champion_trades = test.champion_performance.get('total_trades', 0)
                challenger_trades = test.challenger_performance.get('total_trades', 0)
                
                if (champion_trades >= test.min_sample_size and 
                    challenger_trades >= test.min_sample_size):
                    should_complete = True
                    completion_reason = "Minimum sample size reached"
                
                # Check maximum duration
                if test.start_time and (current_time - test.start_time).days >= self.default_params['max_test_duration_days']:
                    should_complete = True
                    completion_reason = "Maximum test duration reached"
                
                # Check for early statistical significance
                if champion_trades >= 30 and challenger_trades >= 30:
                    significance = await self._calculate_statistical_significance(test)
                    if significance and significance < self.default_params['significance_threshold']:
                        should_complete = True
                        completion_reason = "Early statistical significance detected"
                
                if should_complete:
                    await self._complete_test(test_id, completion_reason)
            
        except Exception as e:
            logger.error(f"Error checking test completion: {str(e)}")
    
    async def _calculate_statistical_significance(self, test: ABTest) -> Optional[float]:
        """Calculate statistical significance between champion and challenger"""
        try:
            # Simple t-test based on return differences (simulated)
            champion_returns = np.random.normal(
                test.champion_performance.get('total_return', 0.15), 0.05, 100
            )
            challenger_returns = np.random.normal(
                test.challenger_performance.get('total_return', 0.18), 0.05, 100
            )
            
            t_stat, p_value = stats.ttest_ind(champion_returns, challenger_returns)
            
            return p_value
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {str(e)}")
            return None
    
    async def _complete_test(self, test_id: str, reason: str):
        """Complete an A/B test"""
        try:
            test = self.active_tests[test_id]
            test.status = TestStatus.COMPLETED
            test.end_time = datetime.now()
            
            # Calculate final results
            significance = await self._calculate_statistical_significance(test)
            test.statistical_significance = significance
            
            # Determine winner
            test.test_result = await self._determine_test_winner(test)
            
            # Update database
            await self._update_test_in_database(test)
            
            # Move to completed tests
            self.completed_tests.append(test)
            del self.active_tests[test_id]
            
            logger.info(f"Test {test_id} completed: {test.test_result.value} ({reason})")
            
        except Exception as e:
            logger.error(f"Error completing test {test_id}: {str(e)}")
    
    async def _determine_test_winner(self, test: ABTest) -> TestResult:
        """Determine the winner of an A/B test"""
        try:
            champion_return = test.champion_performance.get('total_return', 0)
            challenger_return = test.challenger_performance.get('total_return', 0)
            
            # Check statistical significance
            if test.statistical_significance and test.statistical_significance < self.default_params['significance_threshold']:
                if challenger_return > champion_return:
                    return TestResult.CHALLENGER_WINS
                else:
                    return TestResult.CHAMPION_WINS
            else:
                return TestResult.NO_SIGNIFICANT_DIFFERENCE
            
        except Exception as e:
            logger.error(f"Error determining test winner: {str(e)}")
            return TestResult.INSUFFICIENT_DATA
    
    async def _update_test_in_database(self, test: ABTest):
        """Update test record in database"""
        try:
            db_path = "db/ab_testing.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ab_tests SET
                    end_time = ?,
                    status = ?,
                    test_result = ?,
                    statistical_significance = ?
                WHERE test_id = ?
            """, (
                test.end_time,
                test.status.value,
                test.test_result.value if test.test_result else None,
                test.statistical_significance,
                test.test_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating test in database: {str(e)}")
    
    async def _analyze_test_results(self):
        """Analyze and log test results"""
        try:
            if not self.active_tests:
                return
            
            running_tests = [t for t in self.active_tests.values() if t.status == TestStatus.RUNNING]
            
            if running_tests:
                logger.info(f"Active A/B tests: {len(running_tests)}")
                
                for test in running_tests:
                    champion_return = test.champion_performance.get('total_return', 0)
                    challenger_return = test.challenger_performance.get('total_return', 0)
                    
                    performance_diff = challenger_return - champion_return
                    logger.info(f"Test {test.test_id}: Challenger {'ahead' if performance_diff > 0 else 'behind'} by {abs(performance_diff)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Error analyzing test results: {str(e)}")
    
    # Public API methods
    async def create_ab_test(self, name: str, champion_strategy: str, challenger_strategy: str, 
                           allocation_ratio: float = None, min_sample_size: int = None,
                           confidence_level: float = None) -> str:
        """Create a new A/B test"""
        try:
            test_id = f"test_{int(datetime.now().timestamp())}"
            
            test = ABTest(
                test_id=test_id,
                name=name,
                champion_strategy=champion_strategy,
                challenger_strategy=challenger_strategy,
                start_time=datetime.now(),
                end_time=None,
                status=TestStatus.SETUP,
                allocation_ratio=allocation_ratio or self.default_params['challenger_allocation'],
                min_sample_size=min_sample_size or self.default_params['min_sample_size'],
                confidence_level=confidence_level or self.default_params['confidence_level'],
                champion_performance={},
                challenger_performance={},
                statistical_significance=None,
                test_result=None
            )
            
            # Save to database
            await self._save_test_to_database(test)
            
            # Add to active tests
            self.active_tests[test_id] = test
            
            logger.info(f"Created A/B test: {test_id} - {name}")
            
            return test_id
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {str(e)}")
            raise
    
    async def _save_test_to_database(self, test: ABTest):
        """Save test to database"""
        try:
            db_path = "db/ab_testing.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ab_tests 
                (test_id, name, champion_strategy, challenger_strategy, start_time,
                 status, allocation_ratio, min_sample_size, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test.test_id, test.name, test.champion_strategy, test.challenger_strategy,
                test.start_time, test.status.value, test.allocation_ratio,
                test.min_sample_size, test.confidence_level
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving test to database: {str(e)}")
    
    async def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        try:
            if test_id not in self.active_tests:
                return False
            
            test = self.active_tests[test_id]
            test.status = TestStatus.RUNNING
            test.start_time = datetime.now()
            
            await self._update_test_in_database(test)
            
            logger.info(f"Started A/B test: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting test {test_id}: {str(e)}")
            return False
    
    async def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test"""
        try:
            if test_id not in self.active_tests:
                return False
            
            await self._complete_test(test_id, "Manually stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping test {test_id}: {str(e)}")
            return False
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific test"""
        try:
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
            else:
                # Check completed tests
                test = next((t for t in self.completed_tests if t.test_id == test_id), None)
                if not test:
                    return None
            
            return {
                'test_id': test.test_id,
                'name': test.name,
                'status': test.status.value,
                'champion_strategy': test.champion_strategy,
                'challenger_strategy': test.challenger_strategy,
                'champion_performance': test.champion_performance,
                'challenger_performance': test.challenger_performance,
                'statistical_significance': test.statistical_significance,
                'test_result': test.test_result.value if test.test_result else None,
                'start_time': test.start_time,
                'end_time': test.end_time
            }
            
        except Exception as e:
            logger.error(f"Error getting test status: {str(e)}")
            return None
    
    def get_all_tests(self) -> Dict[str, Any]:
        """Get summary of all tests"""
        try:
            return {
                'active_tests': len(self.active_tests),
                'completed_tests': len(self.completed_tests),
                'running_tests': len([t for t in self.active_tests.values() if t.status == TestStatus.RUNNING]),
                'test_summaries': [
                    {
                        'test_id': test.test_id,
                        'name': test.name,
                        'status': test.status.value,
                        'result': test.test_result.value if test.test_result else None
                    }
                    for test in list(self.active_tests.values()) + self.completed_tests
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting all tests: {str(e)}")
            return {}
    
    async def shutdown(self):
        """Shutdown the A/B testing agent"""
        try:
            self.is_initialized = False
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
            logger.info("A/B Testing Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down A/B Testing Agent: {str(e)}")
