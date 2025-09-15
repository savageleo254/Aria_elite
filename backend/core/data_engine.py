import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import aiohttp
import asyncpg
from pathlib import Path

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataEngine:
    """
    Real data engine for fetching market data from various sources
    Replaces mock data generation with actual market data providers
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.session = None
        
    async def initialize(self):
        """Initialize the data engine"""
        try:
            logger.info("Initializing Data Engine")
            
            # Load configuration
            await self._load_config()
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'ARIA-ELITE/1.0'}
            )
            
            # Test database connection
            await self._test_database_connection()
            
            self.is_initialized = True
            logger.info("Data Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Engine: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load data engine configuration"""
        try:
            self.project_config = self.config.load_project_config()
            self.execution_config = self.config.load_execution_config()
            
            # Get data source configuration
            self.data_sources = {
                'mt5': {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 18844,
                    'timeout': 30
                },
                'alpha_vantage': {
                    'enabled': False,  # Disabled by default, requires API key
                    'api_key': None,
                    'base_url': 'https://www.alphavantage.co/query'
                },
                'yahoo_finance': {
                    'enabled': True,
                    'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart'
                },
                'database': {
                    'enabled': True,
                    'connection_string': self.config.get_database_url()
                }
            }
            
            logger.info("Data engine configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load data engine configuration: {str(e)}")
            raise
    
    async def _test_database_connection(self):
        """Test database connection"""
        try:
            if self.data_sources['database']['enabled']:
                conn = await asyncpg.connect(self.data_sources['database']['connection_string'])
                await conn.close()
                logger.info("Database connection test successful")
        except Exception as e:
            logger.warning(f"Database connection test failed: {str(e)}")
    
    async def fetch_training_data(self, symbols: List[str], timeframes: List[str], features: List[str]) -> pd.DataFrame:
        """Fetch real training data from data sources"""
        try:
            logger.info(f"Fetching training data for symbols: {symbols}")
            
            all_data = []
            
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Try to get data from database first
                        data = await self._fetch_from_database(symbol, timeframe)
                        
                        if data is None or data.empty:
                            # Fallback to external API
                            data = await self._fetch_from_api(symbol, timeframe)
                        
                        if data is not None and not data.empty:
                            # Add technical indicators
                            data = self._add_technical_indicators(data, features)
                            data['symbol'] = symbol
                            data['timeframe'] = timeframe
                            all_data.append(data)
                            
                    except Exception as e:
                        logger.error(f"Failed to fetch data for {symbol} {timeframe}: {str(e)}")
                        continue
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.info(f"Successfully fetched {len(combined_data)} records of training data")
                return combined_data
            else:
                raise ValueError("No training data could be fetched from any source")
                
        except Exception as e:
            logger.error(f"Failed to fetch training data: {str(e)}")
            raise
    
    async def _fetch_from_database(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from local database"""
        try:
            if not self.data_sources['database']['enabled']:
                return None
            
            conn = await asyncpg.connect(self.data_sources['database']['connection_string'])
            
            # Calculate date range (last 2 years for training)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data 
                WHERE symbol = $1 AND timeframe = $2 
                AND timestamp BETWEEN $3 AND $4
                ORDER BY timestamp
            """
            
            rows = await conn.fetch(query, symbol, timeframe, start_date, end_date)
            await conn.close()
            
            if rows:
                df = pd.DataFrame(rows)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Database fetch failed for {symbol} {timeframe}: {str(e)}")
            return None
    
    async def _fetch_from_api(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from external API (Yahoo Finance)"""
        try:
            if not self.data_sources['yahoo_finance']['enabled']:
                return None
            
            # Map timeframe to Yahoo Finance interval
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Calculate date range
            end_date = int(datetime.now().timestamp())
            start_date = int((datetime.now() - timedelta(days=730)).timestamp())
            
            url = f"{self.data_sources['yahoo_finance']['base_url']}/{symbol}"
            params = {
                'period1': start_date,
                'period2': end_date,
                'interval': interval,
                'includePrePost': 'false',
                'events': 'div,split'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'chart' in data and 'result' in data['chart']:
                        result = data['chart']['result'][0]
                        timestamps = result['timestamp']
                        quotes = result['indicators']['quote'][0]
                        
                        df = pd.DataFrame({
                            'open': quotes['open'],
                            'high': quotes['high'],
                            'low': quotes['low'],
                            'close': quotes['close'],
                            'volume': quotes['volume']
                        })
                        
                        df['timestamp'] = pd.to_datetime(timestamps, unit='s')
                        df.set_index('timestamp', inplace=True)
                        df.dropna(inplace=True)
                        
                        logger.info(f"Fetched {len(df)} records from Yahoo Finance for {symbol}")
                        return df
                
            return None
            
        except Exception as e:
            logger.error(f"API fetch failed for {symbol} {timeframe}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add technical indicators to the data"""
        try:
            df = data.copy()
            
            # Price-based indicators
            if 'rsi' in features:
                df['rsi'] = self._calculate_rsi(df['close'])
            
            if 'macd' in features:
                macd_line, signal_line, histogram = self._calculate_macd(df['close'])
                df['macd'] = macd_line
                df['macd_signal'] = signal_line
                df['macd_histogram'] = histogram
            
            if 'bb_upper' in features or 'bb_lower' in features:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            if 'volume_sma' in features:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # Price change and volatility
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Trend strength
            df['trend_strength'] = self._calculate_trend_strength(df['close'])
            
            # Support and resistance levels
            df['support_resistance'] = self._calculate_support_resistance(df)
            
            # Target variable for training
            df['target_direction'] = self._calculate_target_direction(df['close'])
            
            # Remove rows with NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {str(e)}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope"""
        def slope(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return 0
            return np.polyfit(x, y, 1)[0]
        
        return prices.rolling(window=period).apply(slope, raw=False)
    
    def _calculate_support_resistance(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate support and resistance levels"""
        highs = df['high'].rolling(window=period).max()
        lows = df['low'].rolling(window=period).min()
        current_price = df['close']
        
        # Distance to nearest support/resistance
        resistance_distance = (highs - current_price) / current_price
        support_distance = (current_price - lows) / current_price
        
        # Return the smaller distance (closer level)
        return np.minimum(resistance_distance, support_distance)
    
    def _calculate_target_direction(self, prices: pd.Series, lookforward: int = 5) -> pd.Series:
        """Calculate target direction for next N periods"""
        future_prices = prices.shift(-lookforward)
        price_change = (future_prices - prices) / prices
        
        # 0 = sell, 1 = hold, 2 = buy
        target = pd.Series(1, index=prices.index)  # Default to hold
        target[price_change > 0.001] = 2  # Buy if > 0.1% increase
        target[price_change < -0.001] = 0  # Sell if > 0.1% decrease
        
        return target
    
    async def fetch_realtime_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time market data"""
        try:
            # Try database first
            data = await self._fetch_from_database(symbol, timeframe)
            
            if data is None or data.empty:
                # Fallback to API
                data = await self._fetch_from_api(symbol, timeframe)
            
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': latest.name.isoformat(),
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': int(latest['volume'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch real-time data for {symbol}: {str(e)}")
            return None
    
    async def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store market data in database"""
        try:
            if not self.data_sources['database']['enabled']:
                return
            
            conn = await asyncpg.connect(self.data_sources['database']['connection_string'])
            
            # Prepare data for insertion
            records = []
            for timestamp, row in data.iterrows():
                records.append((
                    symbol,
                    timeframe,
                    timestamp,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                ))
            
            # Insert data (upsert to handle duplicates)
            query = """
                INSERT INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, timeframe, timestamp) 
                DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """
            
            await conn.executemany(query, records)
            await conn.close()
            
            logger.info(f"Stored {len(records)} records for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Failed to store market data: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
            logger.info("Data Engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during Data Engine cleanup: {str(e)}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'AUDJPY',
            'GBPCAD', 'GBPCHF', 'EURCHF', 'EURAUD', 'EURCAD', 'NZDJPY',
            'AUDNZD', 'GBPNZD', 'CADJPY', 'CHFJPY', 'AUDCHF', 'CADCHF',
            'NZDCHF', 'NZDCAD', 'EURNZD', 'GBPAUD'
        ]
    
    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes"""
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    def get_feature_set(self) -> List[str]:
        """Get list of available features"""
        return [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_sma', 'price_change', 'volatility', 'trend_strength',
            'support_resistance'
        ]
