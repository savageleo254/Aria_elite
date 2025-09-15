import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SignalManager:
    """
    Manages signal generation, position limits, and system state
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_paused = False
        self.active_positions = 0
        self.max_open_positions = 3
        self.signal_history = []
        self.performance_metrics = {
            "total_signals": 0,
            "approved_signals": 0,
            "rejected_signals": 0,
            "executed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        }
        self.last_signal_time = None
        self.poll_interval = 300  # 5 minutes in seconds
        
    async def initialize(self):
        """Initialize the signal manager"""
        try:
            logger.info("Initializing Signal Manager")
            
            # Load configuration
            await self._load_config()
            
            # Start signal generation loop
            asyncio.create_task(self._signal_generation_loop())
            
            # Start performance monitoring
            asyncio.create_task(self._performance_monitoring_loop())
            
            logger.info("Signal Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Signal Manager: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load configuration"""
        try:
            project_config = self.config.load_project_config()
            self.max_open_positions = project_config["execution_policy"]["max_open_positions"]
            self.poll_interval = self._parse_interval(project_config["execution_policy"]["poll_interval"])
            
            logger.info(f"Configuration loaded - Max positions: {self.max_open_positions}, Poll interval: {self.poll_interval}s")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _parse_interval(self, interval_str: str) -> int:
        """Parse interval string to seconds"""
        try:
            if interval_str.endswith('m'):
                return int(interval_str[:-1]) * 60
            elif interval_str.endswith('h'):
                return int(interval_str[:-1]) * 3600
            elif interval_str.endswith('s'):
                return int(interval_str[:-1])
            else:
                return int(interval_str)
        except:
            return 300  # Default to 5 minutes
    
    async def _signal_generation_loop(self):
        """Background task for signal generation"""
        while True:
            try:
                if not self.is_paused:
                    await self._generate_signals()
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _generate_signals(self):
        """Generate signals for configured symbols and strategies"""
        try:
            # This integrates with GeminiWorkflowAgent, SMCModule, AIModelManager, and NewsScraper
        # Real signal generation using multiple strategies
            
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            timeframes = ["H1", "H4"]
            strategies = ["smc_strategy", "ai_strategy"]
            
            for symbol in symbols:
                for timeframe in timeframes:
                    for strategy in strategies:
                        if await self._should_generate_signal(symbol, timeframe, strategy):
                            signal = await self._create_real_signal(symbol, timeframe, strategy)
                            await self._process_signal(signal)
                            
                            # Update last signal time
                            self.last_signal_time = datetime.now()
                            
                            # Update metrics
                            self.performance_metrics["total_signals"] += 1
                            
                            logger.info(f"Generated signal for {symbol} {timeframe} {strategy}")
                
                # Small delay between symbols
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
    
    async def _should_generate_signal(self, symbol: str, timeframe: str, strategy: str) -> bool:
        """Check if should generate signal for given parameters"""
        try:
            # Check if we have max open positions
            if self.active_positions >= self.max_open_positions:
                return False
            
            # Check if we recently generated a signal for this combination
            recent_signals = [
                s for s in self.signal_history
                if s["symbol"] == symbol and s["timeframe"] == timeframe and s["strategy"] == strategy
                and (datetime.now() - s["timestamp"]).total_seconds() < 3600  # 1 hour cooldown
            ]
            
            if len(recent_signals) > 0:
                return False
            
            # Check market hours (simplified)
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 18:  # Avoid overnight trading
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal generation conditions: {str(e)}")
            return False
    
    async def _create_real_signal(self, symbol: str, timeframe: str, strategy: str) -> Dict[str, Any]:
        """Create a real signal based on actual market analysis"""
        try:
            signal_id = f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
        
            # Get real market data
            from .data_engine import DataEngine
            data_engine = DataEngine()
            await data_engine.initialize()
            
            market_data = await data_engine.fetch_realtime_data(symbol, timeframe)
            
            if not market_data:
                logger.warning(f"No real market data available for {symbol}, skipping signal generation")
                return None
            
            current_price = market_data['close']
            
            # Perform real analysis based on strategy
            if strategy == "smc_strategy":
                direction, confidence = await self._analyze_smc_signal(symbol, timeframe, current_price)
            elif strategy == "ai_strategy":
                direction, confidence = await self._analyze_ai_signal(symbol, timeframe, current_price)
            elif strategy == "news_sentiment_strategy":
                direction, confidence = await self._analyze_news_sentiment_signal(symbol, timeframe, current_price)
            else:
                logger.warning(f"Unknown strategy: {strategy}")
                return None
            
            if direction == "HOLD" or confidence < 0.6:
                logger.info(f"Signal not strong enough for {symbol}: {direction} (confidence: {confidence:.2f})")
                return None
            
            # Calculate stop loss and take profit based on ATR
            atr = await self._calculate_atr(symbol, timeframe)
            stop_loss_distance = atr * 2.0  # 2x ATR for stop loss
            take_profit_distance = atr * 3.0  # 3x ATR for take profit
            
            if direction == "BUY":
                stop_loss = current_price - stop_loss_distance
                take_profit = current_price + take_profit_distance
            else:  # SELL
                stop_loss = current_price + stop_loss_distance
                take_profit = current_price - take_profit_distance
            
            signal = {
            "signal_id": signal_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "direction": direction,
            "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
            "timestamp": datetime.now(),
            "status": "pending",
            "market_data": {
                "current_price": current_price,
                    "high_24h": market_data.get('high', current_price),
                    "low_24h": market_data.get('low', current_price),
                    "volume": market_data.get('volume', 0),
                "spread": 0.0001
                },
                "analysis": {
                    "atr": atr,
                    "risk_reward_ratio": take_profit_distance / stop_loss_distance,
                    "strategy_specific": await self._get_strategy_analysis(symbol, timeframe, strategy)
                }
            }
            
            logger.info(f"Created real signal for {symbol}: {direction} at {current_price} (confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error creating real signal for {symbol}: {str(e)}")
            return None
    
    async def _analyze_smc_signal(self, symbol: str, timeframe: str, current_price: float) -> Tuple[str, float]:
        """Analyze SMC strategy signal"""
        try:
            from .smc_module import SMCModule
            smc = SMCModule()
            await smc.initialize()
            
            # Get complete SMC analysis using public method
            smc_analysis = await smc.get_smc_analysis(symbol)
            
            if not smc_analysis or not smc_analysis.get('market_structure'):
                return "HOLD", 0.0
            
            structure = smc_analysis['market_structure'].get('structure', {})
            trading_signals = smc_analysis.get('trading_signals', [])
            
            # Process SMC-generated signals
            if trading_signals:
                best_signal = max(trading_signals, key=lambda x: x.get('confidence', 0))
                if best_signal.get('confidence', 0) > 0.6:
                    direction = "BUY" if best_signal['type'] == 'buy' else "SELL"
                    return direction, best_signal['confidence']
            
            # Fallback: analyze structure directly
            trend = structure.get('trend', 'sideways')
            break_events = structure.get('break_of_structure', [])
            
            if break_events and trend in ['uptrend', 'downtrend']:
                if trend == 'uptrend':
                    return "BUY", 0.7
                elif trend == 'downtrend':
                    return "SELL", 0.7
            
            return "HOLD", 0.3
            
        except Exception as e:
            logger.error(f"Error analyzing SMC signal: {str(e)}")
            return "HOLD", 0.0
    
    async def _analyze_ai_signal(self, symbol: str, timeframe: str, current_price: float) -> Tuple[str, float]:
        """Analyze AI strategy signal"""
        try:
            from models.ai_models import AIModelManager
            ai_manager = AIModelManager()
            await ai_manager.initialize()
            
            # Get recent market data for prediction
            from .data_engine import DataEngine
            data_engine = DataEngine()
            await data_engine.initialize()
            
            data = await data_engine.fetch_training_data([symbol], [timeframe], data_engine.get_feature_set())
            
            if data is None or data.empty:
                logger.warning(f"No training data available for AI signal analysis: {symbol}")
                return "HOLD", 0.0
            
            # Prepare data for prediction
            recent_data = data.tail(50)  # Last 50 periods
            feature_columns = data_engine.get_feature_set()
            
            # Ensure we have the required features
            available_features = [col for col in feature_columns if col in recent_data.columns]
            if not available_features:
                logger.warning(f"No valid features available for AI prediction: {symbol}")
                return "HOLD", 0.0
            
            features = recent_data[available_features].values
            
            # Get individual model predictions and combine them
            predictions = {}
            confidences = {}
            
            # Try different models
            for model_name in ['lightgbm', 'lstm', 'cnn']:
                try:
                    result = await ai_manager.predict(features, model_name=model_name)
                    if result and 'prediction' in result:
                        predictions[model_name] = result['prediction']
                        confidences[model_name] = result.get('confidence', 0.5)
                except Exception as model_error:
                    logger.debug(f"Model {model_name} prediction failed: {str(model_error)}")
                    continue
            
            if not predictions:
                logger.warning(f"No AI model predictions available for {symbol}")
                return "HOLD", 0.0
            
            # Ensemble the predictions
            buy_votes = sum(1 for pred in predictions.values() if pred and pred[-1] == 2)
            sell_votes = sum(1 for pred in predictions.values() if pred and pred[-1] == 0)
            hold_votes = len(predictions) - buy_votes - sell_votes
            
            # Calculate ensemble confidence
            avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
            
            # Determine direction based on majority vote
            if buy_votes > sell_votes and buy_votes > hold_votes:
                return "BUY", min(avg_confidence * (buy_votes / len(predictions)), 0.9)
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                return "SELL", min(avg_confidence * (sell_votes / len(predictions)), 0.9)
            else:
                return "HOLD", avg_confidence * 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing AI signal: {str(e)}")
            return "HOLD", 0.0
    
    async def _analyze_news_sentiment_signal(self, symbol: str, timeframe: str, current_price: float) -> Tuple[str, float]:
        """Analyze news sentiment strategy signal"""
        try:
            from .news_scraper import NewsScraper
            news_scraper = NewsScraper()
            await news_scraper.initialize()
            
            # Get recent news sentiment
            sentiment = await news_scraper.get_sentiment_for_symbol(symbol)
            
            if not sentiment:
                return "HOLD", 0.0
            
            # Determine direction based on sentiment
            sentiment_score = sentiment.get('sentiment_score', 0)
            confidence = abs(sentiment_score)
            
            if sentiment_score > 0.3:
                return "BUY", confidence
            elif sentiment_score < -0.3:
                return "SELL", confidence
            else:
                return "HOLD", confidence
                
        except Exception as e:
            logger.error(f"Error analyzing news sentiment signal: {str(e)}")
            return "HOLD", 0.0
    
    async def _calculate_atr(self, symbol: str, timeframe: str, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            from .data_engine import DataEngine
            data_engine = DataEngine()
            await data_engine.initialize()
            
            data = await data_engine.fetch_training_data([symbol], [timeframe], ['close'])
            
            if data is None or data.empty:
                return 0.001  # Default ATR
            
            symbol_data = data[data['symbol'] == symbol].tail(period * 2)
            
            if symbol_data.empty:
                return 0.001
            
            # Calculate True Range
            high = symbol_data['high']
            low = symbol_data['low']
            close = symbol_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.001
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return 0.001
    
    async def _get_strategy_analysis(self, symbol: str, timeframe: str, strategy: str) -> Dict[str, Any]:
        """Get strategy-specific analysis data"""
        try:
            if strategy == "smc_strategy":
                from .smc_module import SMCModule
                smc = SMCModule()
                await smc.initialize()
                smc_analysis = await smc.get_smc_analysis(symbol)
                return smc_analysis.get('market_structure', {}) if smc_analysis else {}
            
            elif strategy == "ai_strategy":
                from models.ai_models import AIModelManager
                ai_manager = AIModelManager()
                await ai_manager.initialize()
                status = await ai_manager.get_model_status()
                return status
            
            elif strategy == "news_sentiment_strategy":
                from .news_scraper import NewsScraper
                news_scraper = NewsScraper()
                await news_scraper.initialize()
                sentiment = await news_scraper.get_sentiment_for_symbol(symbol)
                return sentiment or {}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting strategy analysis: {str(e)}")
            return {}
    
    async def _process_signal(self, signal: Dict[str, Any]):
        """Process a generated signal"""
        try:
            # Add to history
            self.signal_history.append(signal)
            
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            # Here we would normally send to Gemini for approval
            # For now, we'll simulate approval based on confidence
            if signal["confidence"] > 0.7:
                signal["status"] = "approved"
                self.performance_metrics["approved_signals"] += 1
            else:
                signal["status"] = "rejected"
                self.performance_metrics["rejected_signals"] += 1
                
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            signal["status"] = "error"
    
    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring"""
        while True:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate win rate
            if self.performance_metrics["executed_trades"] > 0:
                self.performance_metrics["win_rate"] = (
                    self.performance_metrics["winning_trades"] / 
                    self.performance_metrics["executed_trades"]
                )
            
            # Clean up old signals (older than 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.signal_history = [
                s for s in self.signal_history
                if s["timestamp"] > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def pause(self):
        """Pause signal generation"""
        try:
            self.is_paused = True
            logger.info("Signal Manager paused")
            
        except Exception as e:
            logger.error(f"Error pausing Signal Manager: {str(e)}")
            raise
    
    async def resume(self):
        """Resume signal generation"""
        try:
            self.is_paused = False
            logger.info("Signal Manager resumed")
            
        except Exception as e:
            logger.error(f"Error resuming Signal Manager: {str(e)}")
            raise
    
    async def update_position_count(self, change: int):
        """Update active position count"""
        try:
            self.active_positions = max(0, self.active_positions + change)
            logger.info(f"Updated position count: {self.active_positions}")
            
        except Exception as e:
            logger.error(f"Error updating position count: {str(e)}")
            raise
    
    async def record_trade_result(self, trade_result: Dict[str, Any]):
        """Record trade result for performance tracking"""
        try:
            self.performance_metrics["executed_trades"] += 1
            
            if trade_result.get("pnl", 0) > 0:
                self.performance_metrics["winning_trades"] += 1
            else:
                self.performance_metrics["losing_trades"] += 1
            
            self.performance_metrics["total_pnl"] += trade_result.get("pnl", 0)
            
            logger.info(f"Recorded trade result: PnL {trade_result.get('pnl', 0)}")
            
        except Exception as e:
            logger.error(f"Error recording trade result: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            return {
                "system_status": "running" if not self.is_paused else "paused",
                "active_positions": self.active_positions,
                "max_positions": self.max_open_positions,
                "daily_pnl": await self._calculate_daily_pnl(),
                "total_trades": self.performance_metrics["executed_trades"],
                "win_rate": self.performance_metrics["win_rate"],
                "last_signal_time": self.last_signal_time,
                "is_paused": self.is_paused,
                "total_signals": self.performance_metrics["total_signals"],
                "approved_signals": self.performance_metrics["approved_signals"],
                "rejected_signals": self.performance_metrics["rejected_signals"]
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            raise
    
    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        try:
            today = datetime.now().date()
            
            # Calculate real daily PnL from trade records
            daily_pnl = await self._calculate_real_daily_pnl(today)
            return daily_pnl
            
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {str(e)}")
            return 0.0
    
    async def _calculate_real_daily_pnl(self, date) -> float:
        """Calculate real daily PnL from trade records"""
        try:
            # This would query the database for actual trade records
            # For now, we'll implement a basic calculation
            
            # Get all trades for the day
            from utils.database import get_database_connection
            
            conn = await get_database_connection()
            
            query = """
                SELECT 
                    SUM(CASE 
                        WHEN direction = 'BUY' THEN (close_price - entry_price) * volume
                        WHEN direction = 'SELL' THEN (entry_price - close_price) * volume
                        ELSE 0
                    END) as daily_pnl
                FROM trades 
                WHERE DATE(close_time) = $1 
                AND status = 'closed'
            """
            
            result = await conn.fetchval(query, date)
            await conn.close()
            
            return float(result) if result is not None else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating real daily PnL: {str(e)}")
            return 0.0
    
    async def get_performance_metrics(self, period: str = "1d") -> Dict[str, Any]:
        """Get performance metrics for specified period"""
        try:
            metrics = self.performance_metrics.copy()
            
            # Add period-specific metrics
            if period == "1d":
                metrics["period_trades"] = self._get_period_trades(1)
            elif period == "7d":
                metrics["period_trades"] = self._get_period_trades(7)
            elif period == "30d":
                metrics["period_trades"] = self._get_period_trades(30)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            raise
    
    async def _get_period_trades(self, days: int) -> int:
        """Get number of trades in specified period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Count real trades from database
            period_trades = await self._count_real_period_trades(cutoff_date)
            return period_trades
            
        except Exception as e:
            logger.error(f"Error getting period trades: {str(e)}")
            return 0
    
    async def _count_real_period_trades(self, cutoff_date) -> int:
        """Count real trades from database"""
        try:
            from utils.database import get_database_connection
            
            conn = await get_database_connection()
            
            query = """
                SELECT COUNT(*) 
                FROM trades 
                WHERE open_time >= $1 
                AND status IN ('open', 'closed')
            """
            
            result = await conn.fetchval(query, cutoff_date)
            await conn.close()
            
            return int(result) if result is not None else 0
            
        except Exception as e:
            logger.error(f"Error counting real period trades: {str(e)}")
            return 0
    
    async def get_signal_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get signal history with pagination"""
        try:
            signals = self.signal_history[-(limit + offset):]
            if offset > 0:
                signals = signals[offset:]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signal history: {str(e)}")
            return []
    
    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get active signals (approved but not yet executed)"""
        try:
            return [s for s in self.signal_history if s["status"] == "approved"]
            
        except Exception as e:
            logger.error(f"Error getting active signals: {str(e)}")
            return []