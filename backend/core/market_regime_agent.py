import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import talib
from scipy import stats

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"  
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class RegimeSignal:
    """Market regime detection signal"""
    regime: MarketRegime
    confidence: float
    timeframe: str
    timestamp: datetime
    indicators: Dict[str, float]
    transition_probability: float = 0.0
    regime_strength: float = 0.0

class MarketRegimeAgent:
    """
    Advanced Market Regime Detection Agent for ARIA-DAN
    Provides real-time market regime classification and transition prediction
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.current_regime = MarketRegime.RANGING
        self.regime_history = []
        self.regime_confidence = 0.0
        self.transition_warnings = []
        
        # Regime detection parameters
        self.timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
        self.lookback_periods = {
            'M1': 100,
            'M5': 200, 
            'M15': 300,
            'H1': 500,
            'H4': 200,
            'D1': 100
        }
        
        # Regime thresholds
        self.thresholds = {
            'trend_strength': 0.7,
            'volatility_threshold': 2.0,
            'crisis_threshold': 3.0,
            'breakout_threshold': 1.5,
            'ranging_threshold': 0.3
        }
        
        # Indicator weights for regime classification
        self.indicator_weights = {
            'adx': 0.25,
            'atr_ratio': 0.20,
            'rsi_divergence': 0.15,
            'bollinger_squeeze': 0.15,
            'volume_profile': 0.10,
            'correlation_break': 0.15
        }
        
        # Regime transition matrix (probabilities)
        self.transition_matrix = self._initialize_transition_matrix()
        
    async def initialize(self):
        """Initialize the market regime agent"""
        try:
            logger.info("Initializing Market Regime Detection Agent")
            
            # Load configuration
            await self._load_regime_config()
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Start regime monitoring loop
            self.monitoring_task = asyncio.create_task(self._regime_monitoring_loop())
            
            self.is_initialized = True
            logger.info("Market Regime Agent initialized - Ready for institutional regime detection")
            
        except Exception as e:
            logger.error(f"Failed to initialize Market Regime Agent: {str(e)}")
            raise
    
    async def _load_regime_config(self):
        """Load regime detection configuration"""
        try:
            # Load custom regime parameters if available
            regime_config = self.config.get('market_regime', {})
            
            # Update thresholds with config values
            if 'thresholds' in regime_config:
                self.thresholds.update(regime_config['thresholds'])
            
            # Update indicator weights
            if 'indicator_weights' in regime_config:
                self.indicator_weights.update(regime_config['indicator_weights'])
            
            logger.info("Market regime configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading regime config: {str(e)}")
    
    async def _initialize_data_sources(self):
        """Initialize market data sources for regime detection"""
        try:
            # Initialize connections to market data feeds
            # This would connect to MT5, Redis, or other data sources
            logger.info("Market regime data sources initialized")
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {str(e)}")
    
    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize regime transition probability matrix"""
        return {
            MarketRegime.TRENDING_BULL: {
                MarketRegime.TRENDING_BULL: 0.85,
                MarketRegime.RANGING: 0.08,
                MarketRegime.VOLATILE: 0.04,
                MarketRegime.REVERSAL: 0.03
            },
            MarketRegime.TRENDING_BEAR: {
                MarketRegime.TRENDING_BEAR: 0.85,
                MarketRegime.RANGING: 0.08,
                MarketRegime.VOLATILE: 0.04,
                MarketRegime.REVERSAL: 0.03
            },
            MarketRegime.RANGING: {
                MarketRegime.RANGING: 0.75,
                MarketRegime.TRENDING_BULL: 0.10,
                MarketRegime.TRENDING_BEAR: 0.10,
                MarketRegime.BREAKOUT: 0.05
            },
            MarketRegime.VOLATILE: {
                MarketRegime.VOLATILE: 0.60,
                MarketRegime.CRISIS: 0.15,
                MarketRegime.RANGING: 0.15,
                MarketRegime.TRENDING_BULL: 0.05,
                MarketRegime.TRENDING_BEAR: 0.05
            },
            MarketRegime.CRISIS: {
                MarketRegime.CRISIS: 0.70,
                MarketRegime.VOLATILE: 0.20,
                MarketRegime.REVERSAL: 0.10
            },
            MarketRegime.BREAKOUT: {
                MarketRegime.TRENDING_BULL: 0.40,
                MarketRegime.TRENDING_BEAR: 0.40,
                MarketRegime.VOLATILE: 0.15,
                MarketRegime.RANGING: 0.05
            },
            MarketRegime.REVERSAL: {
                MarketRegime.TRENDING_BULL: 0.35,
                MarketRegime.TRENDING_BEAR: 0.35,
                MarketRegime.RANGING: 0.20,
                MarketRegime.VOLATILE: 0.10
            }
        }
    
    async def _regime_monitoring_loop(self):
        """Main regime monitoring loop (2-minute cycles)"""
        while self.is_initialized:
            try:
                await self._detect_market_regime()
                await asyncio.sleep(120)  # 2-minute cycles
                
            except Exception as e:
                logger.error(f"Error in regime monitoring loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _detect_market_regime(self):
        """Detect current market regime across all timeframes"""
        try:
            regime_signals = {}
            
            # Analyze each timeframe
            for timeframe in self.timeframes:
                signal = await self._analyze_timeframe_regime(timeframe)
                if signal:
                    regime_signals[timeframe] = signal
            
            # Aggregate signals to determine overall regime
            overall_regime = self._aggregate_regime_signals(regime_signals)
            
            # Check for regime transitions
            transition_probability = self._calculate_transition_probability(overall_regime)
            
            # Update current regime if confidence is high enough
            if overall_regime.confidence > 0.75:
                if overall_regime.regime != self.current_regime:
                    await self._handle_regime_change(overall_regime)
                else:
                    self.regime_confidence = overall_regime.confidence
            
            # Log regime status
            logger.info(f"Current regime: {self.current_regime.value} (confidence: {self.regime_confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
    
    async def _analyze_timeframe_regime(self, timeframe: str) -> Optional[RegimeSignal]:
        """Analyze regime for specific timeframe"""
        try:
            # Get market data for timeframe
            data = await self._get_market_data(timeframe)
            if data is None or len(data) < self.lookback_periods[timeframe]:
                return None
            
            # Calculate regime indicators
            indicators = self._calculate_regime_indicators(data)
            
            # Classify regime based on indicators
            regime, confidence = self._classify_regime(indicators)
            
            # Calculate regime strength
            regime_strength = self._calculate_regime_strength(indicators, regime)
            
            return RegimeSignal(
                regime=regime,
                confidence=confidence,
                timeframe=timeframe,
                timestamp=datetime.now(),
                indicators=indicators,
                regime_strength=regime_strength
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe} regime: {str(e)}")
            return None
    
    async def _get_market_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Get market data for regime analysis"""
        try:
            # This would integrate with actual market data source
            # For now, return None to indicate data not available
            # In production, this would fetch from MT5, Redis, or other sources
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data for {timeframe}: {str(e)}")
            return None
    
    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicators used for regime detection"""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data.get('volume', pd.Series([1] * len(data))).values
            
            indicators = {}
            
            # ADX for trend strength
            adx = talib.ADX(high, low, close, timeperiod=14)
            indicators['adx'] = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
            
            # ATR ratio for volatility
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_ratio = atr[-1] / np.mean(atr[-50:]) if len(atr) > 50 else 1.0
            indicators['atr_ratio'] = float(atr_ratio)
            
            # RSI for momentum divergence
            rsi = talib.RSI(close, timeperiod=14)
            indicators['rsi'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
            
            # Bollinger Bands for squeeze detection
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            bb_squeeze = bb_width < np.percentile([
                (bb_upper[i] - bb_lower[i]) / bb_middle[i] 
                for i in range(max(0, len(bb_upper) - 100), len(bb_upper))
            ], 20)
            indicators['bollinger_squeeze'] = float(bb_squeeze)
            
            # Volume profile indicator
            volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
            volume_ratio = volume[-1] / volume_sma[-1] if not np.isnan(volume_sma[-1]) else 1.0
            indicators['volume_profile'] = float(volume_ratio)
            
            # Price momentum
            momentum = talib.MOM(close, timeperiod=10)
            indicators['momentum'] = float(momentum[-1]) if not np.isnan(momentum[-1]) else 0.0
            
            # Correlation break (placeholder - would use multi-asset data)
            indicators['correlation_break'] = 0.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating regime indicators: {str(e)}")
            return {}
    
    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Classify market regime based on indicators"""
        try:
            regime_scores = {regime: 0.0 for regime in MarketRegime}
            
            # Trending Bull classification
            if indicators.get('adx', 0) > 25 and indicators.get('momentum', 0) > 0:
                regime_scores[MarketRegime.TRENDING_BULL] += 0.4
            
            # Trending Bear classification  
            if indicators.get('adx', 0) > 25 and indicators.get('momentum', 0) < 0:
                regime_scores[MarketRegime.TRENDING_BEAR] += 0.4
            
            # Ranging classification
            if indicators.get('adx', 0) < 20 and indicators.get('bollinger_squeeze', 0) > 0.5:
                regime_scores[MarketRegime.RANGING] += 0.5
            
            # Volatile classification
            if indicators.get('atr_ratio', 1) > self.thresholds['volatility_threshold']:
                regime_scores[MarketRegime.VOLATILE] += 0.4
            
            # Crisis classification
            if indicators.get('atr_ratio', 1) > self.thresholds['crisis_threshold']:
                regime_scores[MarketRegime.CRISIS] += 0.6
            
            # Breakout classification
            if (indicators.get('bollinger_squeeze', 0) < 0.2 and 
                indicators.get('volume_profile', 1) > self.thresholds['breakout_threshold']):
                regime_scores[MarketRegime.BREAKOUT] += 0.5
            
            # Find regime with highest score
            best_regime = max(regime_scores, key=regime_scores.get)
            confidence = min(regime_scores[best_regime], 1.0)
            
            # Default to ranging if confidence is too low
            if confidence < 0.3:
                best_regime = MarketRegime.RANGING
                confidence = 0.5
            
            return best_regime, confidence
            
        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            return MarketRegime.RANGING, 0.5
    
    def _calculate_regime_strength(self, indicators: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate strength of detected regime"""
        try:
            strength = 0.0
            
            if regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
                strength = min(indicators.get('adx', 0) / 50.0, 1.0)
            elif regime == MarketRegime.RANGING:
                strength = 1.0 - min(indicators.get('adx', 0) / 30.0, 1.0)
            elif regime == MarketRegime.VOLATILE:
                strength = min(indicators.get('atr_ratio', 1) / 3.0, 1.0)
            elif regime == MarketRegime.CRISIS:
                strength = min(indicators.get('atr_ratio', 1) / 5.0, 1.0)
            
            return float(strength)
            
        except Exception as e:
            logger.error(f"Error calculating regime strength: {str(e)}")
            return 0.5
    
    def _aggregate_regime_signals(self, regime_signals: Dict[str, RegimeSignal]) -> RegimeSignal:
        """Aggregate regime signals from multiple timeframes"""
        try:
            if not regime_signals:
                return RegimeSignal(
                    regime=MarketRegime.RANGING,
                    confidence=0.5,
                    timeframe='aggregate',
                    timestamp=datetime.now(),
                    indicators={}
                )
            
            # Weight timeframes by importance
            timeframe_weights = {
                'M1': 0.05,
                'M5': 0.10, 
                'M15': 0.20,
                'H1': 0.30,
                'H4': 0.25,
                'D1': 0.10
            }
            
            # Calculate weighted regime votes
            regime_votes = {}
            total_weight = 0.0
            
            for timeframe, signal in regime_signals.items():
                weight = timeframe_weights.get(timeframe, 0.1) * signal.confidence
                
                if signal.regime not in regime_votes:
                    regime_votes[signal.regime] = 0.0
                
                regime_votes[signal.regime] += weight
                total_weight += weight
            
            # Normalize votes
            if total_weight > 0:
                for regime in regime_votes:
                    regime_votes[regime] /= total_weight
            
            # Select regime with highest vote
            best_regime = max(regime_votes, key=regime_votes.get) if regime_votes else MarketRegime.RANGING
            confidence = regime_votes.get(best_regime, 0.5)
            
            # Aggregate indicators
            aggregated_indicators = {}
            for signal in regime_signals.values():
                for key, value in signal.indicators.items():
                    if key not in aggregated_indicators:
                        aggregated_indicators[key] = []
                    aggregated_indicators[key].append(value)
            
            # Average indicators
            final_indicators = {
                key: np.mean(values) for key, values in aggregated_indicators.items()
            }
            
            return RegimeSignal(
                regime=best_regime,
                confidence=confidence,
                timeframe='aggregate',
                timestamp=datetime.now(),
                indicators=final_indicators
            )
            
        except Exception as e:
            logger.error(f"Error aggregating regime signals: {str(e)}")
            return RegimeSignal(
                regime=MarketRegime.RANGING,
                confidence=0.5,
                timeframe='aggregate',
                timestamp=datetime.now(),
                indicators={}
            )
    
    def _calculate_transition_probability(self, regime_signal: RegimeSignal) -> float:
        """Calculate probability of regime transition"""
        try:
            if self.current_regime == regime_signal.regime:
                return 0.0
            
            # Get transition probability from matrix
            transition_prob = self.transition_matrix.get(self.current_regime, {}).get(regime_signal.regime, 0.05)
            
            # Adjust by confidence and regime strength
            adjusted_prob = transition_prob * regime_signal.confidence * regime_signal.regime_strength
            
            return float(adjusted_prob)
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {str(e)}")
            return 0.0
    
    async def _handle_regime_change(self, new_regime_signal: RegimeSignal):
        """Handle regime change detection"""
        try:
            old_regime = self.current_regime
            self.current_regime = new_regime_signal.regime
            self.regime_confidence = new_regime_signal.confidence
            
            # Record regime change
            regime_change = {
                'timestamp': datetime.now(),
                'old_regime': old_regime.value,
                'new_regime': new_regime_signal.regime.value,
                'confidence': new_regime_signal.confidence,
                'indicators': new_regime_signal.indicators
            }
            
            self.regime_history.append(regime_change)
            
            # Keep only last 1000 regime changes
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            logger.warning(f"REGIME CHANGE: {old_regime.value} → {new_regime_signal.regime.value} (confidence: {new_regime_signal.confidence:.2f})")
            
            # Trigger regime change notifications to other systems
            await self._notify_regime_change(regime_change)
            
        except Exception as e:
            logger.error(f"Error handling regime change: {str(e)}")
    
    async def _notify_regime_change(self, regime_change: Dict[str, Any]):
        """Notify other systems of regime change"""
        try:
            # This would notify the autonomous agent and other systems
            # For now, just log the notification
            logger.info(f"Notifying systems of regime change: {regime_change}")
            
        except Exception as e:
            logger.error(f"Error notifying regime change: {str(e)}")
    
    # Public API methods
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current market regime information"""
        return {
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'timestamp': datetime.now(),
            'regime_strength': getattr(self, 'regime_strength', 0.0)
        }
    
    def get_regime_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent regime history"""
        return self.regime_history[-limit:] if self.regime_history else []
    
    def is_regime_stable(self, lookback_minutes: int = 30) -> bool:
        """Check if regime has been stable for given period"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            recent_changes = [
                change for change in self.regime_history 
                if change['timestamp'] > cutoff_time
            ]
            return len(recent_changes) == 0
            
        except Exception as e:
            logger.error(f"Error checking regime stability: {str(e)}")
            return False
    
    def get_regime_recommendation(self) -> Dict[str, Any]:
        """Get trading recommendations based on current regime"""
        try:
            recommendations = {
                'strategy_allocation': {},
                'risk_adjustment': 1.0,
                'position_sizing': 1.0,
                'timeframe_focus': ['H1', 'H4']
            }
            
            if self.current_regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
                recommendations['strategy_allocation'] = {'trend_following': 0.7, 'mean_reversion': 0.1, 'momentum': 0.2}
                recommendations['risk_adjustment'] = 1.2
                recommendations['timeframe_focus'] = ['H1', 'H4', 'D1']
                
            elif self.current_regime == MarketRegime.RANGING:
                recommendations['strategy_allocation'] = {'mean_reversion': 0.6, 'breakout': 0.3, 'grid': 0.1}
                recommendations['risk_adjustment'] = 0.8
                recommendations['timeframe_focus'] = ['M15', 'H1']
                
            elif self.current_regime == MarketRegime.VOLATILE:
                recommendations['strategy_allocation'] = {'volatility': 0.5, 'momentum': 0.3, 'hedge': 0.2}
                recommendations['risk_adjustment'] = 0.6
                recommendations['position_sizing'] = 0.7
                
            elif self.current_regime == MarketRegime.CRISIS:
                recommendations['strategy_allocation'] = {'hedge': 0.8, 'cash': 0.2}
                recommendations['risk_adjustment'] = 0.3
                recommendations['position_sizing'] = 0.3
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting regime recommendation: {str(e)}")
            return {}
    
    async def shutdown(self):
        """Shutdown the market regime agent"""
        try:
            self.is_initialized = False
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
            logger.info("Market Regime Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Market Regime Agent: {str(e)}")
