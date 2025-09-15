import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SMCModule:
    """
    Smart Money Concepts module for analyzing market structure,
    liquidity zones, order blocks, and other SMC patterns
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.market_data_cache = {}
        self.structure_analysis = {}
        self.liquidity_zones = {}
        self.order_blocks = {}
        self.fvg_zones = {}
        self.choch_events = []
        
    async def initialize(self):
        """Initialize the SMC module"""
        try:
            logger.info("Initializing SMC Module")
            
            # Load configuration
            await self._load_config()
            
            # Start analysis loops
            asyncio.create_task(self._market_structure_analysis_loop())
            asyncio.create_task(self._liquidity_analysis_loop())
            asyncio.create_task(self._order_block_analysis_loop())
            
            self.is_initialized = True
            logger.info("SMC Module initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SMC Module: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load SMC configuration"""
        try:
            self.strategy_config = self.config.load_strategy_config()
            self.smc_config = self.strategy_config.get("strategies", {}).get("smc_strategy", {})
            
            logger.info("SMC configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SMC configuration: {str(e)}")
            raise
    
    async def _market_structure_analysis_loop(self):
        """Analyze market structure periodically"""
        while True:
            try:
                await self._analyze_market_structure()
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in market structure analysis loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _liquidity_analysis_loop(self):
        """Analyze liquidity zones periodically"""
        while True:
            try:
                await self._analyze_liquidity_zones()
                await asyncio.sleep(600)  # Analyze every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in liquidity analysis loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _order_block_analysis_loop(self):
        """Analyze order blocks periodically"""
        while True:
            try:
                await self._analyze_order_blocks()
                await asyncio.sleep(900)  # Analyze every 15 minutes
                
            except Exception as e:
                logger.error(f"Error in order block analysis loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _analyze_market_structure(self):
        """Analyze market structure for all symbols"""
        try:
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            
            for symbol in symbols:
                await self._analyze_symbol_structure(symbol)
                
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
    
    async def _analyze_symbol_structure(self, symbol: str):
        """Analyze market structure for a specific symbol"""
        try:
            # Get market data (mock for now)
            market_data = await self._get_market_data(symbol)
            
            # Analyze structure
            structure = await self._identify_structure(market_data)
            
            # Store results
            self.structure_analysis[symbol] = {
                "timestamp": datetime.now(),
                "structure": structure,
                "trend": structure["trend"],
                "key_levels": structure["key_levels"],
                "break_of_structure": structure["break_of_structure"],
                "choch_events": structure["choch_events"]
            }
            
            logger.debug(f"Market structure analyzed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error analyzing symbol structure for {symbol}: {str(e)}")
    
    async def _get_market_data(self, symbol: str, timeframe: str = "H1", periods: int = 100) -> pd.DataFrame:
        """Get real market data for analysis"""
        try:
            # Try to get real data from data engine
            from .data_engine import DataEngine
            data_engine = DataEngine()
            await data_engine.initialize()
            
            # Fetch real market data
            data = await data_engine.fetch_training_data([symbol], [timeframe], data_engine.get_feature_set())
            
            if data is not None and not data.empty:
                # Filter for this symbol and get recent data
                symbol_data = data[data['symbol'] == symbol].tail(periods)
                if not symbol_data.empty:
                    logger.info(f"Using real market data for {symbol}: {len(symbol_data)} records")
                    return symbol_data[['open', 'high', 'low', 'close', 'volume']]
            
            # Fallback to mock data if real data is unavailable
            logger.warning(f"No real data available for {symbol}, using mock data")
            dates = pd.date_range(
                start=datetime.now() - timedelta(hours=periods),
                periods=periods,
                freq='H'
            )

            # Generate realistic price data
            base_price = 1.0850 if symbol == "EURUSD" else 1.2500 if symbol == "GBPUSD" else 110.0
            prices = []
            current_price = base_price

            for i in range(periods):
                # Random walk with trend
                change = np.random.normal(0, 0.001)
                current_price += change

                high = current_price + abs(np.random.normal(0, 0.0005))
                low = current_price - abs(np.random.normal(0, 0.0005))
                open_price = current_price
                close_price = current_price + np.random.normal(0, 0.0003)

                prices.append({
                    'timestamp': dates[i],
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
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            raise
    
    async def _identify_structure(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify market structure"""
        try:
            # Calculate basic indicators
            market_data['sma_20'] = market_data['close'].rolling(window=20).mean()
            market_data['sma_50'] = market_data['close'].rolling(window=50).mean()
            market_data['atr'] = self._calculate_atr(market_data)
            
            # Identify swing highs and lows
            swing_points = self._identify_swing_points(market_data)
            
            # Determine trend
            trend = self._determine_trend(market_data, swing_points)
            
            # Identify break of structure
            break_of_structure = self._identify_break_of_structure(market_data, swing_points, trend)
            
            # Identify CHOCH events
            choch_events = self._identify_choch_events(market_data, swing_points, trend)
            
            # Identify key levels
            key_levels = self._identify_key_levels(swing_points)
            
            return {
                "trend": trend,
                "swing_points": swing_points,
                "break_of_structure": break_of_structure,
                "choch_events": choch_events,
                "key_levels": key_levels,
                "market_data_summary": {
                    "current_price": market_data['close'].iloc[-1],
                    "atr": market_data['atr'].iloc[-1],
                    "volatility": market_data['atr'].iloc[-1] / market_data['close'].iloc[-1]
                }
            }
            
        except Exception as e:
            logger.error(f"Error identifying structure: {str(e)}")
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
    
    def _identify_swing_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List]:
        """Identify swing highs and lows"""
        try:
            swing_highs = []
            swing_lows = []
            
            for i in range(window, len(df) - window):
                # Check for swing high
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                    swing_highs.append({
                        'timestamp': df.index[i],
                        'price': df['high'].iloc[i],
                        'index': i
                    })
                
                # Check for swing low
                if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                    swing_lows.append({
                        'timestamp': df.index[i],
                        'price': df['low'].iloc[i],
                        'index': i
                    })
            
            return {
                'highs': swing_highs,
                'lows': swing_lows
            }
            
        except Exception as e:
            logger.error(f"Error identifying swing points: {str(e)}")
            return {'highs': [], 'lows': []}
    
    def _determine_trend(self, df: pd.DataFrame, swing_points: Dict[str, List]) -> str:
        """Determine market trend"""
        try:
            if len(swing_points['highs']) < 2 or len(swing_points['lows']) < 2:
                return "sideways"
            
            # Check higher highs and higher lows for uptrend
            recent_highs = swing_points['highs'][-3:]
            recent_lows = swing_points['lows'][-3:]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                highs_increasing = all(recent_highs[i]['price'] < recent_highs[i+1]['price'] 
                                     for i in range(len(recent_highs)-1))
                lows_increasing = all(recent_lows[i]['price'] < recent_lows[i+1]['price'] 
                                    for i in range(len(recent_lows)-1))
                
                if highs_increasing and lows_increasing:
                    return "uptrend"
                
                # Check lower highs and lower lows for downtrend
                highs_decreasing = all(recent_highs[i]['price'] > recent_highs[i+1]['price'] 
                                     for i in range(len(recent_highs)-1))
                lows_decreasing = all(recent_lows[i]['price'] > recent_lows[i+1]['price'] 
                                    for i in range(len(recent_lows)-1))
                
                if highs_decreasing and lows_decreasing:
                    return "downtrend"
            
            return "sideways"
            
        except Exception as e:
            logger.error(f"Error determining trend: {str(e)}")
            return "sideways"
    
    def _identify_break_of_structure(self, df: pd.DataFrame, swing_points: Dict[str, List], trend: str) -> List[Dict[str, Any]]:
        """Identify break of structure events"""
        try:
            bos_events = []
            
            if trend == "uptrend" and len(swing_points['highs']) >= 2:
                # Check for break of previous high
                recent_high = swing_points['highs'][-2]
                current_price = df['close'].iloc[-1]
                
                if current_price > recent_high['price']:
                    bos_events.append({
                        'type': 'break_of_high',
                        'broken_level': recent_high['price'],
                        'break_price': current_price,
                        'timestamp': df.index[-1],
                        'strength': 'strong'
                    })
            
            elif trend == "downtrend" and len(swing_points['lows']) >= 2:
                # Check for break of previous low
                recent_low = swing_points['lows'][-2]
                current_price = df['close'].iloc[-1]
                
                if current_price < recent_low['price']:
                    bos_events.append({
                        'type': 'break_of_low',
                        'broken_level': recent_low['price'],
                        'break_price': current_price,
                        'timestamp': df.index[-1],
                        'strength': 'strong'
                    })
            
            return bos_events
            
        except Exception as e:
            logger.error(f"Error identifying break of structure: {str(e)}")
            return []
    
    def _identify_choch_events(self, df: pd.DataFrame, swing_points: Dict[str, List], trend: str) -> List[Dict[str, Any]]:
        """Identify Change of Character (CHOCH) events"""
        try:
            choch_events = []
            
            if len(swing_points['highs']) >= 3 and len(swing_points['lows']) >= 3:
                # Check for failed break of structure
                recent_highs = swing_points['highs'][-3:]
                recent_lows = swing_points['lows'][-3:]
                
                # Failed high break (potential trend reversal to downside)
                if (len(recent_highs) >= 2 and 
                    recent_highs[-2]['price'] > recent_highs[-3]['price'] and
                    recent_highs[-1]['price'] < recent_highs[-2]['price']):
                    
                    choch_events.append({
                        'type': 'failed_high_break',
                        'failed_level': recent_highs[-2]['price'],
                        'timestamp': recent_highs[-1]['timestamp'],
                        'potential_reversal': 'downward'
                    })
                
                # Failed low break (potential trend reversal to upside)
                if (len(recent_lows) >= 2 and 
                    recent_lows[-2]['price'] < recent_lows[-3]['price'] and
                    recent_lows[-1]['price'] > recent_lows[-2]['price']):
                    
                    choch_events.append({
                        'type': 'failed_low_break',
                        'failed_level': recent_lows[-2]['price'],
                        'timestamp': recent_lows[-1]['timestamp'],
                        'potential_reversal': 'upward'
                    })
            
            return choch_events
            
        except Exception as e:
            logger.error(f"Error identifying CHOCH events: {str(e)}")
            return []
    
    def _identify_key_levels(self, swing_points: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify key support and resistance levels"""
        try:
            key_levels = []
            
            # Analyze swing points for key levels
            all_highs = [h['price'] for h in swing_points['highs']]
            all_lows = [l['price'] for l in swing_points['lows']]
            
            # Cluster similar levels
            high_clusters = self._cluster_levels(all_highs, tolerance=0.001)
            low_clusters = self._cluster_levels(all_lows, tolerance=0.001)
            
            # Create resistance levels
            for cluster in high_clusters:
                if len(cluster) >= 2:  # At least 2 touches
                    key_levels.append({
                        'type': 'resistance',
                        'price': np.mean(cluster),
                        'strength': len(cluster),
                        'touches': len(cluster)
                    })
            
            # Create support levels
            for cluster in low_clusters:
                if len(cluster) >= 2:  # At least 2 touches
                    key_levels.append({
                        'type': 'support',
                        'price': np.mean(cluster),
                        'strength': len(cluster),
                        'touches': len(cluster)
                    })
            
            # Sort by strength
            key_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return key_levels[:10]  # Return top 10 levels
            
        except Exception as e:
            logger.error(f"Error identifying key levels: {str(e)}")
            return []
    
    def _cluster_levels(self, levels: List[float], tolerance: float) -> List[List[float]]:
        """Cluster similar price levels"""
        try:
            clusters = []
            
            for level in sorted(levels):
                added = False
                for cluster in clusters:
                    if abs(level - np.mean(cluster)) <= tolerance:
                        cluster.append(level)
                        added = True
                        break
                
                if not added:
                    clusters.append([level])
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering levels: {str(e)}")
            return []
    
    async def _analyze_liquidity_zones(self):
        """Analyze liquidity zones"""
        try:
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            
            for symbol in symbols:
                await self._analyze_symbol_liquidity(symbol)
                
        except Exception as e:
            logger.error(f"Error analyzing liquidity zones: {str(e)}")
    
    async def _analyze_symbol_liquidity(self, symbol: str):
        """Analyze liquidity zones for a specific symbol"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            
            # Identify liquidity zones
            liquidity_zones = await self._identify_liquidity_zones(market_data)
            
            # Store results
            self.liquidity_zones[symbol] = {
                "timestamp": datetime.now(),
                "zones": liquidity_zones
            }
            
            logger.debug(f"Liquidity zones analyzed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error analyzing symbol liquidity for {symbol}: {str(e)}")
    
    async def _identify_liquidity_zones(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify liquidity zones"""
        try:
            liquidity_zones = []
            
            # Identify areas of high volume and price consolidation
            volume_threshold = market_data['volume'].quantile(0.7)
            price_range_threshold = market_data['high'] - market_data['low']
            price_range_threshold = price_range_threshold.quantile(0.3)
            
            # Look for consolidation zones
            for i in range(20, len(market_data) - 20):
                window_data = market_data.iloc[i-20:i+20]
                
                # Check for low volatility (consolidation)
                high_low_range = window_data['high'].max() - window_data['low'].min()
                avg_volume = window_data['volume'].mean()
                
                if (high_low_range <= price_range_threshold and 
                    avg_volume >= volume_threshold):
                    
                    liquidity_zones.append({
                        'type': 'consolidation',
                        'price_range': [window_data['low'].min(), window_data['high'].max()],
                        'center_price': (window_data['low'].min() + window_data['high'].max()) / 2,
                        'strength': avg_volume / volume_threshold,
                        'timestamp': market_data.index[i]
                    })
            
            return liquidity_zones
            
        except Exception as e:
            logger.error(f"Error identifying liquidity zones: {str(e)}")
            return []
    
    async def _analyze_order_blocks(self):
        """Analyze order blocks"""
        try:
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            
            for symbol in symbols:
                await self._analyze_symbol_order_blocks(symbol)
                
        except Exception as e:
            logger.error(f"Error analyzing order blocks: {str(e)}")
    
    async def _analyze_symbol_order_blocks(self, symbol: str):
        """Analyze order blocks for a specific symbol"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            
            # Identify order blocks
            order_blocks = await self._identify_order_blocks(market_data)
            
            # Store results
            self.order_blocks[symbol] = {
                "timestamp": datetime.now(),
                "blocks": order_blocks
            }
            
            logger.debug(f"Order blocks analyzed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error analyzing symbol order blocks for {symbol}: {str(e)}")
    
    async def _identify_order_blocks(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify order blocks"""
        try:
            order_blocks = []
            
            # Identify order blocks based on strong momentum moves
            for i in range(10, len(market_data) - 10):
                # Check for strong bullish move
                if (market_data['close'].iloc[i] > market_data['open'].iloc[i] and
                    market_data['close'].iloc[i] - market_data['open'].iloc[i] > 
                    2 * market_data['atr'].iloc[i]):
                    
                    # Order block is the candle before the strong move
                    ob_candle = market_data.iloc[i-1]
                    
                    order_blocks.append({
                        'type': 'bullish',
                        'price_range': [ob_candle['low'], ob_candle['high']],
                        'center_price': (ob_candle['low'] + ob_candle['high']) / 2,
                        'timestamp': ob_candle.name,
                        'strength': (market_data['close'].iloc[i] - market_data['open'].iloc[i]) / market_data['atr'].iloc[i]
                    })
                
                # Check for strong bearish move
                elif (market_data['close'].iloc[i] < market_data['open'].iloc[i] and
                      market_data['open'].iloc[i] - market_data['close'].iloc[i] > 
                      2 * market_data['atr'].iloc[i]):
                    
                    # Order block is the candle before the strong move
                    ob_candle = market_data.iloc[i-1]
                    
                    order_blocks.append({
                        'type': 'bearish',
                        'price_range': [ob_candle['low'], ob_candle['high']],
                        'center_price': (ob_candle['low'] + ob_candle['high']) / 2,
                        'timestamp': ob_candle.name,
                        'strength': (market_data['open'].iloc[i] - market_data['close'].iloc[i]) / market_data['atr'].iloc[i]
                    })
            
            return order_blocks
            
        except Exception as e:
            logger.error(f"Error identifying order blocks: {str(e)}")
            return []
    
    async def get_smc_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get complete SMC analysis for a symbol"""
        try:
            structure = self.structure_analysis.get(symbol, {})
            liquidity = self.liquidity_zones.get(symbol, {})
            order_blocks = self.order_blocks.get(symbol, {})
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "market_structure": structure,
                "liquidity_zones": liquidity,
                "order_blocks": order_blocks,
                "trading_signals": await self._generate_smc_signals(symbol)
            }
            
        except Exception as e:
            logger.error(f"Error getting SMC analysis for {symbol}: {str(e)}")
            return {}
    
    async def _generate_smc_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate trading signals based on SMC analysis"""
        try:
            signals = []
            structure = self.structure_analysis.get(symbol, {}).get("structure", {})
            
            if not structure:
                return signals
            
            trend = structure.get("trend", "sideways")
            break_of_structure = structure.get("break_of_structure", [])
            choch_events = structure.get("choch_events", [])
            key_levels = structure.get("key_levels", [])
            
            # Generate signals based on break of structure
            for bos in break_of_structure:
                if bos["type"] == "break_of_high" and trend in ["uptrend", "sideways"]:
                    signals.append({
                        "type": "buy",
                        "reason": "break_of_structure_high",
                        "entry_level": bos["break_price"],
                        "confidence": 0.8,
                        "strength": bos["strength"]
                    })
                elif bos["type"] == "break_of_low" and trend in ["downtrend", "sideways"]:
                    signals.append({
                        "type": "sell",
                        "reason": "break_of_structure_low",
                        "entry_level": bos["break_price"],
                        "confidence": 0.8,
                        "strength": bos["strength"]
                    })
            
            # Generate signals based on CHOCH events
            for choch in choch_events:
                if choch["potential_reversal"] == "upward":
                    signals.append({
                        "type": "buy",
                        "reason": "choch_reversal_upward",
                        "entry_level": choch["failed_level"],
                        "confidence": 0.7,
                        "strength": "medium"
                    })
                elif choch["potential_reversal"] == "downward":
                    signals.append({
                        "type": "sell",
                        "reason": "choch_reversal_downward",
                        "entry_level": choch["failed_level"],
                        "confidence": 0.7,
                        "strength": "medium"
                    })
            
            # Generate signals based on key levels
            for level in key_levels:
                if level["type"] == "support" and level["strength"] >= 3:
                    signals.append({
                        "type": "buy",
                        "reason": "key_support_level",
                        "entry_level": level["price"],
                        "confidence": min(0.6 + (level["strength"] * 0.1), 0.9),
                        "strength": level["strength"]
                    })
                elif level["type"] == "resistance" and level["strength"] >= 3:
                    signals.append({
                        "type": "sell",
                        "reason": "key_resistance_level",
                        "entry_level": level["price"],
                        "confidence": min(0.6 + (level["strength"] * 0.1), 0.9),
                        "strength": level["strength"]
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating SMC signals for {symbol}: {str(e)}")
            return []
