import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BacktestingEngine:
    """
    Lightweight local backtesting engine for strategy validation
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.backtest_results = {}
        self.historical_data = {}
        self.strategies = {}
        
    async def initialize(self):
        """Initialize the backtesting engine"""
        try:
            logger.info("Initializing Backtesting Engine")
            
            # Load configuration
            await self._load_config()
            
            # Create data directories
            await self._create_data_directories()
            
            # Load historical data
            await self._load_historical_data()
            
            # Load strategies
            await self._load_strategies()
            
            self.is_initialized = True
            logger.info("Backtesting Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Backtesting Engine: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load backtesting configuration"""
        try:
            self.strategy_config = self.config.load_strategy_config()
            self.project_config = self.config.load_project_config()
            
            logger.info("Backtesting configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load backtesting configuration: {str(e)}")
            raise
    
    async def _create_data_directories(self):
        """Create directories for data storage"""
        try:
            base_dir = Path(__file__).parent.parent.parent
            
            self.data_dirs = {
                "historical": base_dir / "data" / "historical",
                "backtest_results": base_dir / "logs" / "backtest_results"
            }
            
            for dir_path in self.data_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Data directories created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create data directories: {str(e)}")
            raise
    
    async def _load_historical_data(self):
        """Load historical market data"""
        try:
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
            
            for symbol in symbols:
                self.historical_data[symbol] = {}
                
                for timeframe in timeframes:
                    data = await self._load_symbol_data(symbol, timeframe)
                    if data is not None:
                        self.historical_data[symbol][timeframe] = data
            
            logger.info(f"Historical data loaded for {len(self.historical_data)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {str(e)}")
            raise
    
    async def _load_symbol_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load historical data for a specific symbol and timeframe"""
        try:
            # Try to load from file first
            data_file = self.data_dirs["historical"] / f"{symbol}_{timeframe}.parquet"
            
            if data_file.exists():
                data = pd.read_parquet(data_file)
                logger.debug(f"Loaded historical data from file: {symbol}_{timeframe}")
                return data
            else:
                # Fetch real data from data engine if file doesn't exist
                from .data_engine import DataEngine
                data_engine = DataEngine()
                await data_engine.initialize()
                
                data = await data_engine.fetch_training_data([symbol], [timeframe], data_engine.get_feature_set())
                
                if data is not None and not data.empty:
                    # Filter data for this specific symbol and timeframe
                    symbol_data = data[(data['symbol'] == symbol) & (data['timeframe'] == timeframe)]
                    if not symbol_data.empty:
                        # Save to file for future use
                        symbol_data.to_parquet(data_file)
                        logger.debug(f"Fetched and saved real data: {symbol}_{timeframe}")
                        return symbol_data
                
                # Fallback to mock data only if real data is unavailable
                logger.warning(f"No real data available for {symbol}_{timeframe}, using mock data")
                data = await self._generate_mock_data(symbol, timeframe)
                data.to_parquet(data_file)
                logger.debug(f"Generated and saved mock data: {symbol}_{timeframe}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to load symbol data for {symbol}_{timeframe}: {str(e)}")
            return None
    
    async def _generate_mock_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate mock historical data"""
        try:
            # Determine number of periods based on timeframe
            timeframe_periods = {
                "M1": 1440 * 30,    # 30 days of 1-minute data
                "M5": 288 * 30,     # 30 days of 5-minute data
                "M15": 96 * 30,     # 30 days of 15-minute data
                "M30": 48 * 30,     # 30 days of 30-minute data
                "H1": 24 * 90,      # 90 days of 1-hour data
                "H4": 6 * 90,       # 90 days of 4-hour data
                "D1": 365 * 2       # 2 years of daily data
            }
            
            n_periods = timeframe_periods.get(timeframe, 1000)
            
            # Generate timestamps
            end_time = datetime.now()
            if timeframe == "M1":
                freq = "1min"
            elif timeframe == "M5":
                freq = "5min"
            elif timeframe == "M15":
                freq = "15min"
            elif timeframe == "M30":
                freq = "30min"
            elif timeframe == "H1":
                freq = "1H"
            elif timeframe == "H4":
                freq = "4H"
            elif timeframe == "D1":
                freq = "1D"
            else:
                freq = "1H"
            
            timestamps = pd.date_range(
                end=end_time,
                periods=n_periods,
                freq=freq
            )
            
            # Generate base price based on symbol
            base_prices = {
                "EURUSD": 1.0850,
                "GBPUSD": 1.2500,
                "USDJPY": 110.00
            }
            
            base_price = base_prices.get(symbol, 1.0000)
            
            # Generate price data with realistic movements
            prices = []
            current_price = base_price
            
            for i in range(n_periods):
                # Random walk with trend and volatility
                volatility = 0.0001 if timeframe in ["M1", "M5", "M15"] else 0.0005
                trend = 0.00001 * np.sin(i * 0.01)  # Slow oscillating trend
                
                change = np.random.normal(0, volatility) + trend
                current_price += change
                
                # Generate OHLC data
                high = current_price + abs(np.random.normal(0, volatility * 0.5))
                low = current_price - abs(np.random.normal(0, volatility * 0.5))
                open_price = current_price
                close_price = current_price + np.random.normal(0, volatility * 0.3)
                
                # Ensure logical OHLC relationship
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                prices.append({
                    'timestamp': timestamps[i],
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': np.random.randint(100, 1000)
                })
                
                current_price = close_price
            
            df = pd.DataFrame(prices)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to generate mock data for {symbol}_{timeframe}: {str(e)}")
            raise
    
    async def _load_strategies(self):
        """Load trading strategies"""
        try:
            # Load strategy configurations
            strategies_config = self.strategy_config.get("strategies", {})
            
            for strategy_name, strategy_config in strategies_config.items():
                if strategy_config.get("enabled", False):
                    self.strategies[strategy_name] = {
                        "config": strategy_config,
                        "parameters": strategy_config.get("parameters", {})
                    }
            
            logger.info(f"Loaded {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {str(e)}")
            raise
    
    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest for a specific strategy
        
        Args:
            strategy_name: Name of the strategy to test
            symbol: Trading symbol
            timeframe: Chart timeframe
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            parameters: Strategy parameters (optional)
            
        Returns:
            Backtest results
        """
        try:
            logger.info(f"Running backtest: {strategy_name} on {symbol} {timeframe}")
            
            # Validate inputs
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            if symbol not in self.historical_data:
                raise ValueError(f"Historical data not available for {symbol}")
            
            if timeframe not in self.historical_data[symbol]:
                raise ValueError(f"Historical data not available for {symbol} {timeframe}")
            
            # Get historical data
            data = self.historical_data[symbol][timeframe].copy()
            
            # Filter by date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            if len(data) == 0:
                raise ValueError("No data available for specified date range")
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy_name]
            strategy_params = parameters or strategy_config["parameters"]
            
            # Run backtest
            backtest_result = await self._execute_backtest(
                strategy_name,
                data,
                strategy_params,
                symbol,
                timeframe
            )
            
            # Save results
            await self._save_backtest_result(backtest_result)
            
            logger.info(f"Backtest completed: {strategy_name} on {symbol} {timeframe}")
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    async def _execute_backtest(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Execute the backtest"""
        try:
            # Initialize backtest state
            balance = 10000.0  # Starting balance
            equity = balance
            positions = []
            trades = []
            equity_curve = [balance]
            drawdown_curve = [0.0]
            max_drawdown = 0.0
            peak_equity = balance
            
            # Strategy-specific parameters
            if strategy_name == "smc_strategy":
                return await self._backtest_smc_strategy(data, parameters, symbol, timeframe)
            elif strategy_name == "ai_strategy":
                return await self._backtest_ai_strategy(data, parameters, symbol, timeframe)
            elif strategy_name == "news_sentiment_strategy":
                return await self._backtest_news_strategy(data, parameters, symbol, timeframe)
            else:
                # Generic backtest logic
                return await self._backtest_generic_strategy(data, parameters, symbol, timeframe)
                
        except Exception as e:
            logger.error(f"Error executing backtest: {str(e)}")
            raise
    
    async def _backtest_smc_strategy(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Backtest SMC strategy"""
        try:
            # Initialize backtest state
            balance = 10000.0
            equity = balance
            positions = []
            trades = []
            equity_curve = [balance]
            drawdown_curve = [0.0]
            max_drawdown = 0.0
            peak_equity = balance
            
            # Strategy parameters
            min_confidence = parameters.get("min_confidence", 0.75)
            max_risk_per_trade = parameters.get("max_risk_per_trade", 0.02)
            reward_ratio = parameters.get("reward_ratio", 2.0)
            
            # Calculate indicators
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['atr'] = self._calculate_atr(data)
            
            # Generate signals
            signals = []
            for i in range(50, len(data)):
                current_data = data.iloc[i]
                prev_data = data.iloc[i-1]
                
                # Simple SMC-like signal generation
                signal = None
                
                # Break of structure detection
                if (current_data['close'] > data['high'].iloc[i-20:i].max() and 
                    current_data['sma_20'] > current_data['sma_50']):
                    signal = 'BUY'
                elif (current_data['close'] < data['low'].iloc[i-20:i].min() and 
                      current_data['sma_20'] < current_data['sma_50']):
                    signal = 'SELL'
                
                if signal:
                    signals.append({
                        'timestamp': data.index[i],
                        'signal': signal,
                        'price': current_data['close'],
                        'confidence': min_confidence + np.random.uniform(-0.1, 0.1)
                    })
            
            # Execute trades based on signals
            for signal in signals:
                if len(positions) == 0:  # Only one position at a time
                    # Calculate position size
                    risk_amount = balance * max_risk_per_trade
                    atr = data.loc[signal['timestamp'], 'atr']
                    stop_distance = atr * 2.0
                    position_size = risk_amount / stop_distance
                    
                    # Create trade
                    trade = {
                        'entry_time': signal['timestamp'],
                        'entry_price': signal['price'],
                        'direction': signal['signal'],
                        'size': position_size,
                        'stop_loss': signal['price'] - stop_distance if signal['signal'] == 'BUY' else signal['price'] + stop_distance,
                        'take_profit': signal['price'] + (stop_distance * reward_ratio) if signal['signal'] == 'BUY' else signal['price'] - (stop_distance * reward_ratio)
                    }
                    
                    positions.append(trade)
            
            # Simulate trade execution and exit
            for i in range(50, len(data)):
                current_data = data.iloc[i]
                current_time = data.index[i]
                
                # Check open positions
                for position in positions[:]:
                    if position['exit_time'] is None:
                        # Check if stop loss or take profit hit
                        if position['direction'] == 'BUY':
                            if current_data['low'] <= position['stop_loss']:
                                position['exit_time'] = current_time
                                position['exit_price'] = position['stop_loss']
                                position['exit_reason'] = 'stop_loss'
                            elif current_data['high'] >= position['take_profit']:
                                position['exit_time'] = current_time
                                position['exit_price'] = position['take_profit']
                                position['exit_reason'] = 'take_profit'
                        else:  # SELL
                            if current_data['high'] >= position['stop_loss']:
                                position['exit_time'] = current_time
                                position['exit_price'] = position['stop_loss']
                                position['exit_reason'] = 'stop_loss'
                            elif current_data['low'] <= position['take_profit']:
                                position['exit_time'] = current_time
                                position['exit_price'] = position['take_profit']
                                position['exit_reason'] = 'take_profit'
                        
                        # Close trade if exit conditions met
                        if position['exit_time'] is not None:
                            # Calculate P&L
                            if position['direction'] == 'BUY':
                                pnl = (position['exit_price'] - position['entry_price']) * position['size']
                            else:
                                pnl = (position['entry_price'] - position['exit_price']) * position['size']
                            
                            position['pnl'] = pnl
                            position['pnl_percent'] = (pnl / balance) * 100
                            
                            # Update balance
                            balance += pnl
                            trades.append(position)
                            positions.remove(position)
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t['pnl'] for t in trades)
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Calculate equity curve and drawdown
            for trade in trades:
                balance += trade['pnl']
                equity_curve.append(balance)
                
                peak_equity = max(peak_equity, balance)
                drawdown = (peak_equity - balance) / peak_equity
                drawdown_curve.append(drawdown)
                max_drawdown = max(max_drawdown, drawdown)
            
            backtest_result = {
                'strategy_name': 'smc_strategy',
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / 10000.0) * 100,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'final_balance': balance,
                'trades': trades,
                'parameters': parameters,
                'timestamp': datetime.now()
            }
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error backtesting SMC strategy: {str(e)}")
            raise
    
    async def _backtest_ai_strategy(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Backtest AI strategy"""
        try:
            # Mock AI strategy backtest
            # In real implementation, this would use the actual AI models
            
            balance = 10000.0
            trades = []
            
            # Generate mock AI signals
            n_signals = 50
            signal_indices = np.random.choice(50, n_signals, replace=False)
            
            for i in signal_indices:
                if i >= len(data):
                    continue
                
                signal = np.random.choice(['BUY', 'SELL'])
                entry_price = data.iloc[i]['close']
                
                # Calculate trade parameters
                atr = data.iloc[i]['atr'] if 'atr' in data.columns else 0.0010
                stop_distance = atr * 2.0
                reward_ratio = parameters.get("reward_ratio", 2.0)
                
                trade = {
                    'entry_time': data.index[i],
                    'entry_price': entry_price,
                    'direction': signal,
                    'size': 0.1,  # Fixed size for simplicity
                    'stop_loss': entry_price - stop_distance if signal == 'BUY' else entry_price + stop_distance,
                    'take_profit': entry_price + (stop_distance * reward_ratio) if signal == 'BUY' else entry_price - (stop_distance * reward_ratio)
                }
                
                # Simulate random exit
                exit_delay = np.random.randint(10, 100)
                exit_idx = min(i + exit_delay, len(data) - 1)
                
                exit_price = data.iloc[exit_idx]['close']
                
                # Determine exit reason
                if signal == 'BUY':
                    if data.iloc[exit_idx]['low'] <= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'stop_loss'
                    elif data.iloc[exit_idx]['high'] >= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'take_profit'
                    else:
                        trade['exit_price'] = exit_price
                        trade['exit_reason'] = 'manual'
                else:
                    if data.iloc[exit_idx]['high'] >= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'stop_loss'
                    elif data.iloc[exit_idx]['low'] <= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'take_profit'
                    else:
                        trade['exit_price'] = exit_price
                        trade['exit_reason'] = 'manual'
                
                trade['exit_time'] = data.index[exit_idx]
                
                # Calculate P&L
                if trade['direction'] == 'BUY':
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['size']
                else:
                    trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['size']
                
                trade['pnl_percent'] = (trade['pnl'] / balance) * 100
                trades.append(trade)
                balance += trade['pnl']
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in trades)
            
            backtest_result = {
                'strategy_name': 'ai_strategy',
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / 10000.0) * 100,
                'final_balance': balance,
                'trades': trades,
                'parameters': parameters,
                'timestamp': datetime.now()
            }
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error backtesting AI strategy: {str(e)}")
            raise
    
    async def _backtest_news_strategy(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Backtest news sentiment strategy"""
        try:
            # Mock news sentiment strategy
            balance = 10000.0
            trades = []
            
            # Generate mock news events
            n_events = 30
            event_indices = np.random.choice(50, n_events, replace=False)
            
            for i in event_indices:
                if i >= len(data):
                    continue
                
                # Random sentiment
                sentiment = np.random.choice(['positive', 'negative', 'neutral'])
                sentiment_strength = np.random.uniform(0.3, 0.9)
                
                # Generate signal based on sentiment
                if sentiment == 'positive' and sentiment_strength > 0.6:
                    signal = 'BUY'
                elif sentiment == 'negative' and sentiment_strength > 0.6:
                    signal = 'SELL'
                else:
                    continue
                
                entry_price = data.iloc[i]['close']
                
                # Calculate trade parameters
                atr = data.iloc[i]['atr'] if 'atr' in data.columns else 0.0010
                stop_distance = atr * 1.5
                
                trade = {
                    'entry_time': data.index[i],
                    'entry_price': entry_price,
                    'direction': signal,
                    'size': 0.1,
                    'stop_loss': entry_price - stop_distance if signal == 'BUY' else entry_price + stop_distance,
                    'take_profit': entry_price + (stop_distance * 1.5) if signal == 'BUY' else entry_price - (stop_distance * 1.5),
                    'sentiment': sentiment,
                    'sentiment_strength': sentiment_strength
                }
                
                # Simulate exit
                exit_delay = np.random.randint(5, 50)
                exit_idx = min(i + exit_delay, len(data) - 1)
                
                exit_price = data.iloc[exit_idx]['close']
                trade['exit_price'] = exit_price
                trade['exit_time'] = data.index[exit_idx]
                trade['exit_reason'] = 'time_based'
                
                # Calculate P&L
                if trade['direction'] == 'BUY':
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['size']
                else:
                    trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['size']
                
                trade['pnl_percent'] = (trade['pnl'] / balance) * 100
                trades.append(trade)
                balance += trade['pnl']
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in trades)
            
            backtest_result = {
                'strategy_name': 'news_sentiment_strategy',
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / 10000.0) * 100,
                'final_balance': balance,
                'trades': trades,
                'parameters': parameters,
                'timestamp': datetime.now()
            }
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error backtesting news strategy: {str(e)}")
            raise
    
    async def _backtest_generic_strategy(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Backtest generic strategy"""
        try:
            # Simple moving average crossover strategy
            balance = 10000.0
            trades = []
            
            # Calculate indicators
            data['sma_fast'] = data['close'].rolling(window=10).mean()
            data['sma_slow'] = data['close'].rolling(window=30).mean()
            
            # Generate signals
            for i in range(30, len(data)):
                if data['sma_fast'].iloc[i-1] <= data['sma_slow'].iloc[i-1] and data['sma_fast'].iloc[i] > data['sma_slow'].iloc[i]:
                    # Buy signal
                    if len(trades) == 0 or trades[-1]['direction'] != 'BUY':
                        trade = {
                            'entry_time': data.index[i],
                            'entry_price': data.iloc[i]['close'],
                            'direction': 'BUY',
                            'size': 0.1
                        }
                        trades.append(trade)
                
                elif data['sma_fast'].iloc[i-1] >= data['sma_slow'].iloc[i-1] and data['sma_fast'].iloc[i] < data['sma_slow'].iloc[i]:
                    # Sell signal
                    if len(trades) == 0 or trades[-1]['direction'] != 'SELL':
                        trade = {
                            'entry_time': data.index[i],
                            'entry_price': data.iloc[i]['close'],
                            'direction': 'SELL',
                            'size': 0.1
                        }
                        trades.append(trade)
            
            # Close trades
            for i in range(len(trades)):
                if i < len(trades) - 1:
                    trades[i]['exit_time'] = trades[i+1]['entry_time']
                    trades[i]['exit_price'] = trades[i+1]['entry_price']
                else:
                    trades[i]['exit_time'] = data.index[-1]
                    trades[i]['exit_price'] = data.iloc[-1]['close']
                
                trades[i]['exit_reason'] = 'signal'
                
                # Calculate P&L
                if trades[i]['direction'] == 'BUY':
                    trades[i]['pnl'] = (trades[i]['exit_price'] - trades[i]['entry_price']) * trades[i]['size']
                else:
                    trades[i]['pnl'] = (trades[i]['entry_price'] - trades[i]['exit_price']) * trades[i]['size']
                
                trades[i]['pnl_percent'] = (trades[i]['pnl'] / balance) * 100
                balance += trades[i]['pnl']
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in trades)
            
            backtest_result = {
                'strategy_name': 'generic_strategy',
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / 10000.0) * 100,
                'final_balance': balance,
                'trades': trades,
                'parameters': parameters,
                'timestamp': datetime.now()
            }
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error backtesting generic strategy: {str(e)}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(0, index=df.index)
    
    async def _save_backtest_result(self, result: Dict[str, Any]):
        """Save backtest result"""
        try:
            # Generate unique ID
            result_id = f"backtest_{result['strategy_name']}_{result['symbol']}_{result['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result['id'] = result_id
            
            # Store in memory
            self.backtest_results[result_id] = result
            
            # Save to file
            result_file = self.data_dirs["backtest_results"] / f"{result_id}.json"
            
            # Convert datetime objects to strings for JSON serialization
            result_copy = result.copy()
            if 'start_date' in result_copy and hasattr(result_copy['start_date'], 'isoformat'):
                result_copy['start_date'] = result_copy['start_date'].isoformat()
            if 'end_date' in result_copy and hasattr(result_copy['end_date'], 'isoformat'):
                result_copy['end_date'] = result_copy['end_date'].isoformat()
            
            # Convert trade timestamps
            for trade in result_copy.get('trades', []):
                if 'entry_time' in trade and hasattr(trade['entry_time'], 'isoformat'):
                    trade['entry_time'] = trade['entry_time'].isoformat()
                if 'exit_time' in trade and hasattr(trade['exit_time'], 'isoformat'):
                    trade['exit_time'] = trade['exit_time'].isoformat()
            
            with open(result_file, 'w') as f:
                json.dump(result_copy, f, indent=2)
            
            logger.info(f"Backtest result saved: {result_id}")
            
        except Exception as e:
            logger.error(f"Error saving backtest result: {str(e)}")
    
    async def get_backtest_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get backtest result by ID"""
        try:
            if result_id in self.backtest_results:
                return self.backtest_results[result_id]
            
            # Try to load from file
            result_file = self.data_dirs["backtest_results"] / f"{result_id}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                self.backtest_results[result_id] = result
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting backtest result {result_id}: {str(e)}")
            return None
    
    async def get_backtest_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get backtest history"""
        try:
            results = list(self.backtest_results.values())
            
            # Sort by timestamp
            results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error getting backtest history: {str(e)}")
            return []
    
    async def compare_strategies(
        self,
        strategies: List[str],
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Compare multiple strategies"""
        try:
            comparison_results = {}
            
            for strategy_name in strategies:
                try:
                    result = await self.run_backtest(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    comparison_results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Error backtesting strategy {strategy_name}: {str(e)}")
                    comparison_results[strategy_name] = {"error": str(e)}
            
            # Generate comparison summary
            summary = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "strategies_tested": len(strategies),
                "strategies_completed": len([r for r in comparison_results.values() if "error" not in r]),
                "results": comparison_results,
                "best_strategy": None,
                "worst_strategy": None,
                "timestamp": datetime.now()
            }
            
            # Find best and worst strategies
            completed_results = {k: v for k, v in comparison_results.items() if "error" not in v}
            if completed_results:
                best_strategy = max(completed_results.items(), key=lambda x: x[1].get('total_pnl', 0))
                worst_strategy = min(completed_results.items(), key=lambda x: x[1].get('total_pnl', 0))
                summary["best_strategy"] = best_strategy[0]
                summary["worst_strategy"] = worst_strategy[0]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            raise
    
    async def optimize_strategy(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        parameter_ranges: Dict[str, List[Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        try:
            logger.info(f"Optimizing strategy {strategy_name} for {symbol} {timeframe}")
            
            # Generate parameter combinations
            import itertools
            
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            param_combinations = list(itertools.product(*param_values))
            
            optimization_results = []
            
            # Run backtest for each parameter combination
            for i, combination in enumerate(param_combinations):
                try:
                    parameters = dict(zip(param_names, combination))
                    
                    result = await self.run_backtest(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        parameters=parameters
                    )
                    
                    optimization_results.append({
                        "parameters": parameters,
                        "result": result,
                        "fitness": result.get('total_pnl', 0)  # Fitness function
                    })
                    
                    logger.info(f"Optimization progress: {i+1}/{len(param_combinations)}")
                    
                except Exception as e:
                    logger.error(f"Error optimizing with parameters {combination}: {str(e)}")
                    continue
            
            # Sort by fitness
            optimization_results.sort(key=lambda x: x["fitness"], reverse=True)
            
            # Generate optimization summary
            summary = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "parameter_ranges": parameter_ranges,
                "total_combinations": len(param_combinations),
                "completed_combinations": len(optimization_results),
                "best_parameters": optimization_results[0]["parameters"] if optimization_results else None,
                "best_fitness": optimization_results[0]["fitness"] if optimization_results else None,
                "best_result": optimization_results[0]["result"] if optimization_results else None,
                "all_results": optimization_results,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Strategy optimization completed for {strategy_name}")
            return summary
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            raise
