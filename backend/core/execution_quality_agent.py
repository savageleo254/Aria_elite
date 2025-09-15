import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3
from statistics import mean, stdev

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class ExecutionQuality(Enum):
    """Execution quality classification"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    symbol: str
    order_id: str
    execution_time: datetime
    requested_price: float
    filled_price: float
    slippage_pips: float
    slippage_cost: float
    fill_time_ms: int
    partial_fills: int
    broker: str
    execution_quality: ExecutionQuality
    market_impact: float

class ExecutionQualityAgent:
    """
    Execution Quality Analysis Agent for ARIA-DAN
    Monitors trade execution quality and optimizes order routing
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.execution_history = []
        self.broker_performance = {}
        self.quality_alerts = []
        
        # Quality thresholds (pips)
        self.quality_thresholds = {
            'excellent': 0.5,
            'good': 1.0,
            'fair': 2.0,
            'poor': 3.0,
            'critical': 5.0
        }
        
        # Performance targets
        self.performance_targets = {
            'max_slippage_pips': 1.5,
            'max_fill_time_ms': 500,
            'min_fill_rate': 0.98,
            'max_rejection_rate': 0.02
        }
        
        # Broker comparison data
        self.broker_stats = {}
        
    async def initialize(self):
        """Initialize the execution quality agent"""
        try:
            logger.info("Initializing Execution Quality Agent")
            
            await self._load_execution_config()
            await self._initialize_execution_database()
            await self._load_historical_performance()
            
            # Start monitoring loop
            self.monitoring_task = asyncio.create_task(self._execution_monitoring_loop())
            
            self.is_initialized = True
            logger.info("Execution Quality Agent initialized - Ready for trade execution analysis")
            
        except Exception as e:
            logger.error(f"Failed to initialize Execution Quality Agent: {str(e)}")
            raise
    
    async def _load_execution_config(self):
        """Load execution quality configuration"""
        try:
            exec_config = self.config.get('execution_quality', {})
            
            if 'quality_thresholds' in exec_config:
                self.quality_thresholds.update(exec_config['quality_thresholds'])
            
            if 'performance_targets' in exec_config:
                self.performance_targets.update(exec_config['performance_targets'])
            
            logger.info("Execution quality configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading execution config: {str(e)}")
    
    async def _initialize_execution_database(self):
        """Initialize database for execution tracking"""
        try:
            db_path = "db/execution_quality.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    execution_time DATETIME,
                    requested_price REAL,
                    filled_price REAL,
                    slippage_pips REAL,
                    slippage_cost REAL,
                    fill_time_ms INTEGER,
                    partial_fills INTEGER,
                    broker TEXT,
                    execution_quality TEXT,
                    market_impact REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    broker TEXT,
                    symbol TEXT,
                    metrics TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Execution quality database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing execution database: {str(e)}")
    
    async def _load_historical_performance(self):
        """Load historical execution performance data"""
        try:
            db_path = "db/execution_quality.db"
            conn = sqlite3.connect(db_path)
            
            # Load recent execution data
            query = """
                SELECT * FROM executions 
                WHERE execution_time > datetime('now', '-30 days')
                ORDER BY execution_time DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) > 0:
                # Calculate broker performance stats
                self._calculate_broker_performance(df)
                logger.info(f"Loaded {len(df)} historical executions")
            else:
                logger.info("No historical execution data found")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading historical performance: {str(e)}")
    
    def _calculate_broker_performance(self, df: pd.DataFrame):
        """Calculate broker performance statistics"""
        try:
            for broker in df['broker'].unique():
                broker_data = df[df['broker'] == broker]
                
                self.broker_stats[broker] = {
                    'avg_slippage': broker_data['slippage_pips'].mean(),
                    'avg_fill_time': broker_data['fill_time_ms'].mean(),
                    'execution_count': len(broker_data),
                    'quality_distribution': broker_data['execution_quality'].value_counts().to_dict(),
                    'last_updated': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Error calculating broker performance: {str(e)}")
    
    async def _execution_monitoring_loop(self):
        """Main execution quality monitoring loop"""
        while self.is_initialized:
            try:
                await self._analyze_recent_executions()
                await self._check_performance_alerts()
                await self._optimize_routing_decisions()
                
                await asyncio.sleep(300)  # 5-minute cycles
                
            except Exception as e:
                logger.error(f"Error in execution monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _analyze_recent_executions(self):
        """Analyze recent execution quality"""
        try:
            # Get executions from last hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_executions = [
                exec for exec in self.execution_history 
                if exec.execution_time > cutoff_time
            ]
            
            if recent_executions:
                # Analyze quality trends
                quality_analysis = self._analyze_quality_trends(recent_executions)
                
                # Check for deteriorating performance
                if quality_analysis.get('quality_declining', False):
                    await self._generate_quality_alert(quality_analysis)
                
                logger.info(f"Analyzed {len(recent_executions)} recent executions")
            
        except Exception as e:
            logger.error(f"Error analyzing recent executions: {str(e)}")
    
    def _analyze_quality_trends(self, executions: List[ExecutionMetrics]) -> Dict[str, Any]:
        """Analyze execution quality trends"""
        try:
            if len(executions) < 5:
                return {'insufficient_data': True}
            
            # Calculate metrics
            avg_slippage = mean([e.slippage_pips for e in executions])
            avg_fill_time = mean([e.fill_time_ms for e in executions])
            poor_executions = len([e for e in executions if e.execution_quality in [ExecutionQuality.POOR, ExecutionQuality.CRITICAL]])
            
            # Compare with targets
            quality_declining = (
                avg_slippage > self.performance_targets['max_slippage_pips'] or
                avg_fill_time > self.performance_targets['max_fill_time_ms'] or
                poor_executions / len(executions) > 0.1
            )
            
            return {
                'avg_slippage': avg_slippage,
                'avg_fill_time': avg_fill_time,
                'poor_execution_rate': poor_executions / len(executions),
                'quality_declining': quality_declining,
                'sample_count': len(executions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing quality trends: {str(e)}")
            return {}
    
    async def _generate_quality_alert(self, analysis: Dict[str, Any]):
        """Generate execution quality alert"""
        try:
            alert = {
                'alert_type': 'execution_quality_decline',
                'severity': 'warning',
                'message': f"Execution quality declining: avg_slippage={analysis.get('avg_slippage', 0):.2f} pips",
                'metrics': analysis,
                'timestamp': datetime.now()
            }
            
            self.quality_alerts.append(alert)
            
            # Log to database
            await self._log_quality_alert(alert)
            
            logger.warning(f"EXECUTION QUALITY ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error generating quality alert: {str(e)}")
    
    async def _log_quality_alert(self, alert: Dict[str, Any]):
        """Log quality alert to database"""
        try:
            db_path = "db/execution_quality.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO quality_alerts 
                (alert_type, severity, message, metrics)
                VALUES (?, ?, ?, ?)
            """, (
                alert['alert_type'],
                alert['severity'],
                alert['message'],
                str(alert['metrics'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging quality alert: {str(e)}")
    
    async def _check_performance_alerts(self):
        """Check for performance threshold violations"""
        try:
            if not self.broker_stats:
                return
            
            for broker, stats in self.broker_stats.items():
                # Check slippage threshold
                if stats['avg_slippage'] > self.performance_targets['max_slippage_pips']:
                    await self._generate_broker_alert(broker, 'high_slippage', stats)
                
                # Check fill time threshold
                if stats['avg_fill_time'] > self.performance_targets['max_fill_time_ms']:
                    await self._generate_broker_alert(broker, 'slow_fills', stats)
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {str(e)}")
    
    async def _generate_broker_alert(self, broker: str, alert_type: str, stats: Dict[str, Any]):
        """Generate broker-specific performance alert"""
        try:
            alert = {
                'alert_type': alert_type,
                'severity': 'warning',
                'broker': broker,
                'message': f"Broker {broker} {alert_type}: {stats}",
                'timestamp': datetime.now()
            }
            
            self.quality_alerts.append(alert)
            logger.warning(f"BROKER ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error generating broker alert: {str(e)}")
    
    async def _optimize_routing_decisions(self):
        """Optimize order routing based on performance data"""
        try:
            if len(self.broker_stats) < 2:
                return
            
            # Rank brokers by performance
            broker_rankings = self._rank_brokers_by_performance()
            
            # Update routing preferences
            await self._update_routing_preferences(broker_rankings)
            
        except Exception as e:
            logger.error(f"Error optimizing routing decisions: {str(e)}")
    
    def _rank_brokers_by_performance(self) -> List[Tuple[str, float]]:
        """Rank brokers by overall performance score"""
        try:
            broker_scores = []
            
            for broker, stats in self.broker_stats.items():
                # Calculate composite score (lower is better)
                slippage_score = stats['avg_slippage'] / self.performance_targets['max_slippage_pips']
                fill_time_score = stats['avg_fill_time'] / self.performance_targets['max_fill_time_ms']
                
                composite_score = (slippage_score * 0.6) + (fill_time_score * 0.4)
                broker_scores.append((broker, composite_score))
            
            # Sort by score (ascending - lower is better)
            return sorted(broker_scores, key=lambda x: x[1])
            
        except Exception as e:
            logger.error(f"Error ranking brokers: {str(e)}")
            return []
    
    async def _update_routing_preferences(self, broker_rankings: List[Tuple[str, float]]):
        """Update order routing preferences based on performance"""
        try:
            if broker_rankings:
                best_broker = broker_rankings[0][0]
                logger.info(f"Best performing broker: {best_broker}")
                
                # This would integrate with actual order routing system
                # For now, just log the recommendation
                
        except Exception as e:
            logger.error(f"Error updating routing preferences: {str(e)}")
    
    # Public API methods
    async def record_execution(self, execution: ExecutionMetrics):
        """Record a trade execution for quality analysis"""
        try:
            # Add to history
            self.execution_history.append(execution)
            
            # Keep only last 10000 executions
            if len(self.execution_history) > 10000:
                self.execution_history = self.execution_history[-10000:]
            
            # Log to database
            await self._log_execution(execution)
            
            # Update broker stats
            await self._update_broker_stats(execution)
            
        except Exception as e:
            logger.error(f"Error recording execution: {str(e)}")
    
    async def _log_execution(self, execution: ExecutionMetrics):
        """Log execution to database"""
        try:
            db_path = "db/execution_quality.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO executions 
                (symbol, order_id, execution_time, requested_price, filled_price,
                 slippage_pips, slippage_cost, fill_time_ms, partial_fills,
                 broker, execution_quality, market_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.symbol,
                execution.order_id,
                execution.execution_time,
                execution.requested_price,
                execution.filled_price,
                execution.slippage_pips,
                execution.slippage_cost,
                execution.fill_time_ms,
                execution.partial_fills,
                execution.broker,
                execution.execution_quality.value,
                execution.market_impact
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging execution: {str(e)}")
    
    async def _update_broker_stats(self, execution: ExecutionMetrics):
        """Update broker performance statistics"""
        try:
            broker = execution.broker
            
            if broker not in self.broker_stats:
                self.broker_stats[broker] = {
                    'executions': [],
                    'avg_slippage': 0.0,
                    'avg_fill_time': 0.0,
                    'execution_count': 0,
                    'quality_distribution': {},
                    'last_updated': datetime.now()
                }
            
            # Add execution to broker stats
            stats = self.broker_stats[broker]
            stats['executions'].append(execution)
            stats['execution_count'] = len(stats['executions'])
            
            # Recalculate averages
            stats['avg_slippage'] = mean([e.slippage_pips for e in stats['executions']])
            stats['avg_fill_time'] = mean([e.fill_time_ms for e in stats['executions']])
            stats['last_updated'] = datetime.now()
            
            # Keep only last 1000 executions per broker
            if len(stats['executions']) > 1000:
                stats['executions'] = stats['executions'][-1000:]
            
        except Exception as e:
            logger.error(f"Error updating broker stats: {str(e)}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution quality summary"""
        try:
            if not self.execution_history:
                return {'no_data': True}
            
            recent_executions = self.execution_history[-100:]  # Last 100 executions
            
            return {
                'total_executions': len(self.execution_history),
                'recent_executions': len(recent_executions),
                'avg_slippage': mean([e.slippage_pips for e in recent_executions]),
                'avg_fill_time': mean([e.fill_time_ms for e in recent_executions]),
                'quality_distribution': {
                    quality.value: len([e for e in recent_executions if e.execution_quality == quality])
                    for quality in ExecutionQuality
                },
                'broker_count': len(self.broker_stats),
                'alert_count': len(self.quality_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting execution summary: {str(e)}")
            return {}
    
    def get_broker_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get broker performance comparison"""
        return self.broker_stats.copy()
    
    def get_routing_recommendation(self, symbol: str = None) -> Dict[str, Any]:
        """Get order routing recommendation"""
        try:
            if not self.broker_stats:
                return {'recommendation': 'insufficient_data'}
            
            rankings = self._rank_brokers_by_performance()
            
            if rankings:
                return {
                    'recommended_broker': rankings[0][0],
                    'performance_score': rankings[0][1],
                    'alternative_brokers': rankings[1:3],  # Top 3 alternatives
                    'reasoning': 'Based on recent execution quality metrics'
                }
            
            return {'recommendation': 'no_ranking_available'}
            
        except Exception as e:
            logger.error(f"Error getting routing recommendation: {str(e)}")
            return {}
    
    async def shutdown(self):
        """Shutdown the execution quality agent"""
        try:
            self.is_initialized = False
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
            logger.info("Execution Quality Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Execution Quality Agent: {str(e)}")
