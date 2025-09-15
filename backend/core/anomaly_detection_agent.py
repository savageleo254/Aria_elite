import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class AnomalyType(Enum):
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge" 
    CORRELATION_BREAK = "correlation_break"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_DRY_UP = "liquidity_dry_up"
    UNUSUAL_SPREAD = "unusual_spread"
    PATTERN_DEVIATION = "pattern_deviation"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyAlert:
    timestamp: datetime
    symbol: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float
    description: str
    metrics: Dict[str, float]
    recommended_action: str

class AnomalyDetectionAgent:
    """
    Advanced Anomaly Detection System for ARIA-DAN
    Real-time detection of market anomalies and flash events
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.anomaly_alerts = []
        self.detection_models = {}
        self.baseline_metrics = {}
        
        # Detection thresholds
        self.thresholds = {
            'price_spike_std': 3.0,        # 3 standard deviations
            'volume_surge_multiplier': 5.0, # 5x average volume
            'correlation_break_threshold': 0.3, # 30% correlation drop
            'flash_crash_drop': 0.02,      # 2% drop in seconds
            'spread_anomaly_multiplier': 3.0, # 3x normal spread
            'confidence_threshold': 0.7     # Minimum confidence for alerts
        }
        
        # Historical data windows
        self.lookback_periods = {
            'price_baseline': 100,
            'volume_baseline': 50, 
            'correlation_window': 200,
            'pattern_memory': 500
        }
        
    async def initialize(self):
        """Initialize the anomaly detection agent"""
        try:
            logger.info("Initializing Anomaly Detection Agent")
            
            await self._load_anomaly_config()
            await self._initialize_detection_models()
            await self._load_baseline_metrics()
            
            # Start detection loops
            self.price_task = asyncio.create_task(self._price_anomaly_loop())
            self.volume_task = asyncio.create_task(self._volume_anomaly_loop())
            self.correlation_task = asyncio.create_task(self._correlation_anomaly_loop())
            self.flash_task = asyncio.create_task(self._flash_event_loop())
            
            self.is_initialized = True
            logger.info("Anomaly Detection Agent initialized - Ready for market anomaly detection")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anomaly Detection Agent: {str(e)}")
            raise
    
    async def _load_anomaly_config(self):
        """Load anomaly detection configuration"""
        try:
            anomaly_config = self.config.get('anomaly_detection', {})
            
            if 'thresholds' in anomaly_config:
                self.thresholds.update(anomaly_config['thresholds'])
            
            if 'lookback_periods' in anomaly_config:
                self.lookback_periods.update(anomaly_config['lookback_periods'])
            
            logger.info("Anomaly detection configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading anomaly config: {str(e)}")
    
    async def _initialize_detection_models(self):
        """Initialize ML models for anomaly detection"""
        try:
            # Initialize Isolation Forest for each anomaly type
            self.detection_models = {
                AnomalyType.PRICE_SPIKE: IsolationForest(contamination=0.1, random_state=42),
                AnomalyType.VOLUME_SURGE: IsolationForest(contamination=0.05, random_state=42),
                AnomalyType.PATTERN_DEVIATION: IsolationForest(contamination=0.15, random_state=42)
            }
            
            # Initialize scalers
            self.scalers = {
                anomaly_type: StandardScaler() 
                for anomaly_type in self.detection_models.keys()
            }
            
            logger.info("Anomaly detection models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing detection models: {str(e)}")
    
    async def _load_baseline_metrics(self):
        """Load or calculate baseline metrics for each symbol"""
        try:
            # Simulate baseline metrics loading
            symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
            
            for symbol in symbols:
                self.baseline_metrics[symbol] = {
                    'price_std': np.random.uniform(0.01, 0.05),
                    'volume_mean': np.random.uniform(1000, 5000),
                    'volume_std': np.random.uniform(200, 1000),
                    'spread_mean': np.random.uniform(0.0001, 0.0005),
                    'spread_std': np.random.uniform(0.00005, 0.0002),
                    'last_updated': datetime.now()
                }
            
            logger.info(f"Loaded baseline metrics for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading baseline metrics: {str(e)}")
    
    async def _price_anomaly_loop(self):
        """Monitor for price anomalies (1-minute cycles)"""
        while self.is_initialized:
            try:
                await self._detect_price_anomalies()
                await asyncio.sleep(60)  # 1-minute cycles
                
            except Exception as e:
                logger.error(f"Error in price anomaly loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _volume_anomaly_loop(self):
        """Monitor for volume anomalies (2-minute cycles)"""
        while self.is_initialized:
            try:
                await self._detect_volume_anomalies()
                await asyncio.sleep(120)  # 2-minute cycles
                
            except Exception as e:
                logger.error(f"Error in volume anomaly loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _correlation_anomaly_loop(self):
        """Monitor for correlation breaks (5-minute cycles)"""
        while self.is_initialized:
            try:
                await self._detect_correlation_anomalies()
                await asyncio.sleep(300)  # 5-minute cycles
                
            except Exception as e:
                logger.error(f"Error in correlation anomaly loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _flash_event_loop(self):
        """Monitor for flash events (real-time)"""
        while self.is_initialized:
            try:
                await self._detect_flash_events()
                await asyncio.sleep(5)  # 5-second cycles for flash detection
                
            except Exception as e:
                logger.error(f"Error in flash event loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _detect_price_anomalies(self):
        """Detect price spike anomalies"""
        try:
            symbols = list(self.baseline_metrics.keys())
            
            for symbol in symbols:
                # Simulate getting recent price data
                price_data = await self._get_recent_prices(symbol)
                
                if len(price_data) < 10:
                    continue
                
                # Calculate price changes
                price_changes = np.diff(price_data)
                recent_change = price_changes[-1]
                
                # Get baseline volatility
                baseline = self.baseline_metrics[symbol]
                expected_std = baseline['price_std']
                
                # Detect spikes using z-score
                z_score = abs(recent_change) / expected_std
                
                if z_score > self.thresholds['price_spike_std']:
                    severity = self._calculate_price_spike_severity(z_score)
                    
                    alert = AnomalyAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        anomaly_type=AnomalyType.PRICE_SPIKE,
                        severity=severity,
                        confidence=min(z_score / 5.0, 1.0),
                        description=f"Price spike detected: {z_score:.2f} standard deviations",
                        metrics={'z_score': z_score, 'price_change': recent_change},
                        recommended_action=self._get_spike_recommendation(severity)
                    )
                    
                    await self._process_anomaly_alert(alert)
            
        except Exception as e:
            logger.error(f"Error detecting price anomalies: {str(e)}")
    
    async def _get_recent_prices(self, symbol: str) -> List[float]:
        """Get recent price data (simulated)"""
        # Simulate price data with occasional spikes
        base_price = 2000.0 if symbol == 'XAUUSD' else 1.1
        prices = []
        
        for i in range(20):
            # Add normal variation
            normal_change = np.random.normal(0, 1) * 0.01
            # Occasionally add a spike
            if np.random.random() < 0.05:  # 5% chance of spike
                spike = np.random.normal(0, 1) * 0.05  # Larger spike
                normal_change += spike
            
            base_price *= (1 + normal_change)
            prices.append(base_price)
        
        return prices
    
    def _calculate_price_spike_severity(self, z_score: float) -> SeverityLevel:
        """Calculate severity level for price spikes"""
        if z_score > 6.0:
            return SeverityLevel.CRITICAL
        elif z_score > 4.5:
            return SeverityLevel.HIGH
        elif z_score > 3.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _get_spike_recommendation(self, severity: SeverityLevel) -> str:
        """Get recommendation for price spike"""
        recommendations = {
            SeverityLevel.CRITICAL: "EMERGENCY: Halt trading, investigate market conditions",
            SeverityLevel.HIGH: "Reduce position sizes, increase monitoring",
            SeverityLevel.MEDIUM: "Exercise caution, verify news events",
            SeverityLevel.LOW: "Monitor closely, normal operations"
        }
        return recommendations.get(severity, "Monitor situation")
    
    async def _detect_volume_anomalies(self):
        """Detect volume surge anomalies"""
        try:
            symbols = list(self.baseline_metrics.keys())
            
            for symbol in symbols:
                # Simulate getting recent volume data
                volume_data = await self._get_recent_volumes(symbol)
                
                if len(volume_data) < 5:
                    continue
                
                current_volume = volume_data[-1]
                baseline = self.baseline_metrics[symbol]
                
                volume_multiplier = current_volume / baseline['volume_mean']
                
                if volume_multiplier > self.thresholds['volume_surge_multiplier']:
                    severity = self._calculate_volume_severity(volume_multiplier)
                    
                    alert = AnomalyAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        anomaly_type=AnomalyType.VOLUME_SURGE,
                        severity=severity,
                        confidence=min(volume_multiplier / 10.0, 1.0),
                        description=f"Volume surge: {volume_multiplier:.1f}x normal volume",
                        metrics={'volume_multiplier': volume_multiplier, 'current_volume': current_volume},
                        recommended_action=self._get_volume_recommendation(severity)
                    )
                    
                    await self._process_anomaly_alert(alert)
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {str(e)}")
    
    async def _get_recent_volumes(self, symbol: str) -> List[float]:
        """Get recent volume data (simulated)"""
        baseline_volume = self.baseline_metrics[symbol]['volume_mean']
        volumes = []
        
        for i in range(10):
            # Normal volume variation
            volume = baseline_volume * np.random.uniform(0.7, 1.3)
            # Occasionally add volume surge
            if np.random.random() < 0.03:  # 3% chance of surge
                volume *= np.random.uniform(3, 8)  # 3-8x surge
            volumes.append(volume)
        
        return volumes
    
    def _calculate_volume_severity(self, multiplier: float) -> SeverityLevel:
        """Calculate severity for volume surges"""
        if multiplier > 15.0:
            return SeverityLevel.CRITICAL
        elif multiplier > 10.0:
            return SeverityLevel.HIGH
        elif multiplier > 7.0:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _get_volume_recommendation(self, severity: SeverityLevel) -> str:
        """Get recommendation for volume surge"""
        recommendations = {
            SeverityLevel.CRITICAL: "Major market event likely, investigate immediately",
            SeverityLevel.HIGH: "Significant activity, check news and adjust strategies",
            SeverityLevel.MEDIUM: "Increased activity, monitor for opportunities",
            SeverityLevel.LOW: "Elevated volume, normal monitoring"
        }
        return recommendations.get(severity, "Monitor volume trends")
    
    async def _detect_correlation_anomalies(self):
        """Detect correlation breaks between assets"""
        try:
            # Simulate correlation analysis between major pairs
            correlations = await self._calculate_asset_correlations()
            
            for pair, current_corr in correlations.items():
                expected_corr = await self._get_expected_correlation(pair)
                
                correlation_drop = expected_corr - abs(current_corr)
                
                if correlation_drop > self.thresholds['correlation_break_threshold']:
                    severity = self._calculate_correlation_severity(correlation_drop)
                    
                    alert = AnomalyAlert(
                        timestamp=datetime.now(),
                        symbol=pair,
                        anomaly_type=AnomalyType.CORRELATION_BREAK,
                        severity=severity,
                        confidence=correlation_drop / 0.8,  # Normalize to 0-1
                        description=f"Correlation break: {correlation_drop:.2f} drop from expected",
                        metrics={'correlation_drop': correlation_drop, 'current_correlation': current_corr},
                        recommended_action="Review cross-asset strategies and hedging positions"
                    )
                    
                    await self._process_anomaly_alert(alert)
            
        except Exception as e:
            logger.error(f"Error detecting correlation anomalies: {str(e)}")
    
    async def _calculate_asset_correlations(self) -> Dict[str, float]:
        """Calculate current asset correlations (simulated)"""
        pairs = ['EURUSD-GBPUSD', 'XAUUSD-SILVER', 'EURUSD-XAUUSD']
        correlations = {}
        
        for pair in pairs:
            # Simulate correlation with occasional breaks
            normal_corr = np.random.uniform(0.6, 0.9)
            if np.random.random() < 0.1:  # 10% chance of correlation break
                normal_corr *= np.random.uniform(0.2, 0.7)  # Reduce correlation
            correlations[pair] = normal_corr
        
        return correlations
    
    async def _get_expected_correlation(self, pair: str) -> float:
        """Get expected correlation for asset pair"""
        # Simulate expected correlations
        expected_correlations = {
            'EURUSD-GBPUSD': 0.85,
            'XAUUSD-SILVER': 0.75,
            'EURUSD-XAUUSD': 0.65
        }
        return expected_correlations.get(pair, 0.7)
    
    def _calculate_correlation_severity(self, drop: float) -> SeverityLevel:
        """Calculate severity for correlation breaks"""
        if drop > 0.6:
            return SeverityLevel.CRITICAL
        elif drop > 0.5:
            return SeverityLevel.HIGH
        elif drop > 0.4:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    async def _detect_flash_events(self):
        """Detect flash crash/spike events"""
        try:
            symbols = list(self.baseline_metrics.keys())
            
            for symbol in symbols:
                # Get very recent price data (last few seconds)
                flash_data = await self._get_flash_price_data(symbol)
                
                if len(flash_data) < 3:
                    continue
                
                # Calculate rapid price change
                price_change = (flash_data[-1] - flash_data[0]) / flash_data[0]
                
                if abs(price_change) > self.thresholds['flash_crash_drop']:
                    event_type = AnomalyType.FLASH_CRASH
                    severity = SeverityLevel.CRITICAL
                    
                    alert = AnomalyAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        anomaly_type=event_type,
                        severity=severity,
                        confidence=min(abs(price_change) / 0.05, 1.0),
                        description=f"Flash event: {price_change*100:.1f}% move in seconds",
                        metrics={'price_change_pct': price_change * 100},
                        recommended_action="IMMEDIATE: Halt trading, activate emergency protocols"
                    )
                    
                    await self._process_anomaly_alert(alert)
            
        except Exception as e:
            logger.error(f"Error detecting flash events: {str(e)}")
    
    async def _get_flash_price_data(self, symbol: str) -> List[float]:
        """Get flash-frequency price data (simulated)"""
        base_price = 2000.0 if symbol == 'XAUUSD' else 1.1
        prices = [base_price]
        
        # Simulate 5 seconds of data
        for i in range(4):
            # Normal small change
            change = np.random.normal(0, 0.0001)
            # Very rare flash event
            if np.random.random() < 0.001:  # 0.1% chance
                change += np.random.choice([-0.03, 0.03])  # ±3% flash
            
            base_price *= (1 + change)
            prices.append(base_price)
        
        return prices
    
    async def _process_anomaly_alert(self, alert: AnomalyAlert):
        """Process and log anomaly alert"""
        try:
            # Add to alert history
            self.anomaly_alerts.append(alert)
            
            # Keep only last 1000 alerts
            if len(self.anomaly_alerts) > 1000:
                self.anomaly_alerts = self.anomaly_alerts[-1000:]
            
            # Log alert based on severity
            if alert.severity == SeverityLevel.CRITICAL:
                logger.critical(f"CRITICAL ANOMALY: {alert.symbol} - {alert.description}")
            elif alert.severity == SeverityLevel.HIGH:
                logger.error(f"HIGH ANOMALY: {alert.symbol} - {alert.description}")
            else:
                logger.warning(f"ANOMALY: {alert.symbol} - {alert.description}")
            
            # Trigger emergency actions for critical alerts
            if alert.severity == SeverityLevel.CRITICAL:
                await self._trigger_emergency_response(alert)
            
        except Exception as e:
            logger.error(f"Error processing anomaly alert: {str(e)}")
    
    async def _trigger_emergency_response(self, alert: AnomalyAlert):
        """Trigger emergency response for critical anomalies"""
        try:
            logger.critical(f"TRIGGERING EMERGENCY RESPONSE: {alert.anomaly_type.value} - {alert.symbol}")
            
            # This would integrate with risk management system
            # For now, just log the emergency response
            
        except Exception as e:
            logger.error(f"Error triggering emergency response: {str(e)}")
    
    # Public API methods
    def get_anomaly_status(self) -> Dict[str, Any]:
        """Get current anomaly detection status"""
        try:
            recent_alerts = [a for a in self.anomaly_alerts if a.timestamp > datetime.now() - timedelta(hours=1)]
            
            severity_counts = {severity.value: 0 for severity in SeverityLevel}
            for alert in recent_alerts:
                severity_counts[alert.severity.value] += 1
            
            return {
                'total_alerts': len(self.anomaly_alerts),
                'recent_alerts_1h': len(recent_alerts),
                'severity_distribution': severity_counts,
                'monitored_symbols': len(self.baseline_metrics),
                'detection_models_active': len(self.detection_models),
                'last_update': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly status: {str(e)}")
            return {}
    
    def get_recent_alerts(self, hours: int = 24, severity: SeverityLevel = None) -> List[AnomalyAlert]:
        """Get recent anomaly alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [a for a in self.anomaly_alerts if a.timestamp > cutoff_time]
            
            if severity:
                recent_alerts = [a for a in recent_alerts if a.severity == severity]
            
            return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {str(e)}")
            return []
    
    def get_symbol_anomaly_history(self, symbol: str, days: int = 7) -> List[AnomalyAlert]:
        """Get anomaly history for specific symbol"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            symbol_alerts = [
                a for a in self.anomaly_alerts 
                if a.symbol == symbol and a.timestamp > cutoff_time
            ]
            
            return sorted(symbol_alerts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting symbol anomaly history: {str(e)}")
            return []
    
    async def update_baseline_metrics(self, symbol: str, metrics: Dict[str, float]):
        """Update baseline metrics for symbol"""
        try:
            if symbol not in self.baseline_metrics:
                self.baseline_metrics[symbol] = {}
            
            self.baseline_metrics[symbol].update(metrics)
            self.baseline_metrics[symbol]['last_updated'] = datetime.now()
            
            logger.info(f"Updated baseline metrics for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating baseline metrics: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the anomaly detection agent"""
        try:
            self.is_initialized = False
            
            # Cancel monitoring tasks
            for task in [getattr(self, f'{task_name}_task', None) for task_name in ['price', 'volume', 'correlation', 'flash']]:
                if task:
                    task.cancel()
            
            logger.info("Anomaly Detection Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Anomaly Detection Agent: {str(e)}")
