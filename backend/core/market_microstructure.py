import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict, deque
import redis.asyncio as redis
from decimal import Decimal

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class TickData:
    """Tick-level market data"""
    timestamp: datetime
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: int
    side: str  # 'buy' or 'sell'

@dataclass
class OrderFlowMetrics:
    """Order flow analysis metrics"""
    timestamp: datetime
    symbol: str
    buy_volume: int
    sell_volume: int
    imbalance_ratio: float
    vwap: Decimal
    twap: Decimal
    participation_rate: float
    market_impact: float

@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics"""
    timestamp: datetime
    symbol: str
    bid_ask_spread: Decimal
    spread_bps: float
    market_depth: Dict[str, int]
    liquidity_score: float
    slippage_estimate: float

class MarketMicrostructure:
    """
    Institutional-grade market microstructure analysis system
    Provides order flow, liquidity, and execution quality analytics
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.redis_client = None
        
        # Market data storage - optimized for 8GB RAM
        self.tick_buffer = defaultdict(lambda: deque(maxlen=2000))  # Reduced from 10K to 2K
        self.order_flow_cache = defaultdict(dict)
        self.liquidity_cache = defaultdict(dict)
        
        # Analysis parameters - optimized for CPU processing
        self.analysis_window = 120  # Reduced to 2 minutes for faster processing
        self.tick_aggregation_window = 2  # Increased to 2 seconds to reduce CPU load
        
        # Memory management
        self.memory_efficient_mode = True
        self.max_memory_usage_mb = 1024  # Limit to 1GB RAM usage
        
    async def initialize(self):
        """Initialize the market microstructure module"""
        try:
            logger.info("Initializing Market Microstructure Module")
            
            # Initialize Redis for real-time data
            self.redis_client = await aioredis.create_redis_pool(
                'redis://localhost', encoding='utf-8'
            )
            
            # Start analysis loops
            asyncio.create_task(self._tick_processing_loop())
            asyncio.create_task(self._order_flow_analysis_loop())
            asyncio.create_task(self._liquidity_analysis_loop())
            
            self.is_initialized = True
            logger.info("Market Microstructure Module initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Market Microstructure Module: {str(e)}")
            raise
    
    async def process_tick(self, tick: TickData):
        """Process incoming tick data"""
        try:
            # Store tick in buffer
            self.tick_buffer[tick.symbol].append(tick)
            
            # Stream to Redis for real-time analysis
            await self._stream_tick_to_redis(tick)
            
            # Update real-time metrics
            await self._update_realtime_metrics(tick)
            
        except Exception as e:
            logger.error(f"Error processing tick for {tick.symbol}: {str(e)}")
    
    async def _stream_tick_to_redis(self, tick: TickData):
        """Stream tick data to Redis"""
        try:
            tick_data = {
                'timestamp': tick.timestamp.isoformat(),
                'symbol': tick.symbol,
                'bid': str(tick.bid),
                'ask': str(tick.ask),
                'last': str(tick.last),
                'volume': tick.volume,
                'side': tick.side
            }
            
            # Publish to Redis stream
            await self.redis_client.publish(f'ticks:{tick.symbol}', str(tick_data))
            
        except Exception as e:
            logger.error(f"Error streaming tick to Redis: {str(e)}")
    
    async def _tick_processing_loop(self):
        """Process tick data continuously"""
        while True:
            try:
                for symbol in self.tick_buffer.keys():
                    await self._analyze_tick_patterns(symbol)
                
                await asyncio.sleep(self.tick_aggregation_window)
                
            except Exception as e:
                logger.error(f"Error in tick processing loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def _analyze_tick_patterns(self, symbol: str):
        """Analyze tick patterns for microstructure signals"""
        try:
            ticks = list(self.tick_buffer[symbol])
            if len(ticks) < 100:
                return
            
            recent_ticks = ticks[-100:]  # Last 100 ticks
            
            # Calculate tick-level metrics
            tick_imbalance = self._calculate_tick_imbalance(recent_ticks)
            price_impact = self._calculate_price_impact(recent_ticks)
            clustering = self._detect_price_clustering(recent_ticks)
            
            # Store analysis results
            analysis = {
                'timestamp': datetime.now(),
                'tick_imbalance': tick_imbalance,
                'price_impact': price_impact,
                'clustering_score': clustering,
                'tick_count': len(recent_ticks)
            }
            
            await self._store_tick_analysis(symbol, analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing tick patterns for {symbol}: {str(e)}")
    
    def _calculate_tick_imbalance(self, ticks: List[TickData]) -> float:
        """Calculate order flow imbalance from tick data"""
        try:
            buy_ticks = sum(1 for tick in ticks if tick.side == 'buy')
            sell_ticks = sum(1 for tick in ticks if tick.side == 'sell')
            total_ticks = len(ticks)
            
            if total_ticks == 0:
                return 0.0
            
            return (buy_ticks - sell_ticks) / total_ticks
            
        except Exception as e:
            logger.error(f"Error calculating tick imbalance: {str(e)}")
            return 0.0
    
    def _calculate_price_impact(self, ticks: List[TickData]) -> float:
        """Calculate price impact from tick sequence"""
        try:
            if len(ticks) < 2:
                return 0.0
            
            # Calculate cumulative price impact
            impacts = []
            for i in range(1, len(ticks)):
                prev_mid = (ticks[i-1].bid + ticks[i-1].ask) / 2
                curr_mid = (ticks[i].bid + ticks[i].ask) / 2
                impact = float((curr_mid - prev_mid) / prev_mid)
                impacts.append(abs(impact))
            
            return np.mean(impacts) if impacts else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {str(e)}")
            return 0.0
    
    def _detect_price_clustering(self, ticks: List[TickData]) -> float:
        """Detect price clustering patterns"""
        try:
            if len(ticks) < 10:
                return 0.0
            
            # Extract last prices and convert to pips
            prices = [float(tick.last) for tick in ticks]
            
            # Calculate price distribution
            price_counts = defaultdict(int)
            for price in prices:
                # Round to nearest pip
                rounded_price = round(price, 5)
                price_counts[rounded_price] += 1
            
            # Calculate clustering coefficient
            total_prices = len(prices)
            unique_prices = len(price_counts)
            
            return 1.0 - (unique_prices / total_prices) if total_prices > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error detecting price clustering: {str(e)}")
            return 0.0
    
    async def _order_flow_analysis_loop(self):
        """Analyze order flow continuously"""
        while True:
            try:
                symbols = list(self.tick_buffer.keys())
                
                for symbol in symbols:
                    await self._analyze_order_flow(symbol)
                
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in order flow analysis loop: {str(e)}")
                await asyncio.sleep(5)
    
    async def _analyze_order_flow(self, symbol: str):
        """Comprehensive order flow analysis"""
        try:
            ticks = list(self.tick_buffer[symbol])
            if len(ticks) < 50:
                return
            
            # Analyze recent tick data
            analysis_window_ticks = ticks[-300:]  # Last 300 ticks (~5 minutes)
            
            # Calculate order flow metrics
            buy_volume = sum(tick.volume for tick in analysis_window_ticks if tick.side == 'buy')
            sell_volume = sum(tick.volume for tick in analysis_window_ticks if tick.side == 'sell')
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return
            
            imbalance_ratio = (buy_volume - sell_volume) / total_volume
            
            # Calculate VWAP and TWAP
            vwap = self._calculate_vwap(analysis_window_ticks)
            twap = self._calculate_twap(analysis_window_ticks)
            
            # Market impact estimation
            market_impact = self._estimate_market_impact(analysis_window_ticks)
            
            # Participation rate (volume as % of total market)
            participation_rate = self._calculate_participation_rate(analysis_window_ticks)
            
            # Create order flow metrics
            metrics = OrderFlowMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                imbalance_ratio=imbalance_ratio,
                vwap=vwap,
                twap=twap,
                participation_rate=participation_rate,
                market_impact=market_impact
            )
            
            # Store metrics
            self.order_flow_cache[symbol] = metrics
            await self._store_order_flow_metrics(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing order flow for {symbol}: {str(e)}")
    
    def _calculate_vwap(self, ticks: List[TickData]) -> Decimal:
        """Calculate Volume Weighted Average Price"""
        try:
            total_value = sum(tick.last * tick.volume for tick in ticks)
            total_volume = sum(tick.volume for tick in ticks)
            
            return Decimal(str(total_value / total_volume)) if total_volume > 0 else Decimal('0')
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return Decimal('0')
    
    def _calculate_twap(self, ticks: List[TickData]) -> Decimal:
        """Calculate Time Weighted Average Price"""
        try:
            if not ticks:
                return Decimal('0')
            
            total_price = sum(tick.last for tick in ticks)
            return Decimal(str(total_price / len(ticks)))
            
        except Exception as e:
            logger.error(f"Error calculating TWAP: {str(e)}")
            return Decimal('0')
    
    def _estimate_market_impact(self, ticks: List[TickData]) -> float:
        """Estimate market impact of order flow"""
        try:
            if len(ticks) < 10:
                return 0.0
            
            # Simple market impact model based on volume and price movement
            total_volume = sum(tick.volume for tick in ticks)
            price_start = float(ticks[0].last)
            price_end = float(ticks[-1].last)
            
            price_change = abs(price_end - price_start) / price_start
            volume_impact = total_volume / 1000000  # Normalize volume
            
            return price_change * volume_impact
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {str(e)}")
            return 0.0
    
    def _calculate_participation_rate(self, ticks: List[TickData]) -> float:
        """Calculate participation rate"""
        try:
            # Mock implementation - in production, compare against total market volume
            total_volume = sum(tick.volume for tick in ticks)
            estimated_market_volume = total_volume * 10  # Assume we see 10% of market
            
            return min(total_volume / estimated_market_volume, 1.0) if estimated_market_volume > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating participation rate: {str(e)}")
            return 0.0
    
    async def _liquidity_analysis_loop(self):
        """Analyze liquidity continuously"""
        while True:
            try:
                symbols = list(self.tick_buffer.keys())
                
                for symbol in symbols:
                    await self._analyze_liquidity(symbol)
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"Error in liquidity analysis loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _analyze_liquidity(self, symbol: str):
        """Comprehensive liquidity analysis"""
        try:
            ticks = list(self.tick_buffer[symbol])
            if len(ticks) < 20:
                return
            
            recent_ticks = ticks[-100:]  # Last 100 ticks
            
            # Calculate bid-ask spread metrics
            spreads = [(tick.ask - tick.bid) for tick in recent_ticks]
            avg_spread = sum(spreads) / len(spreads)
            
            # Convert to basis points
            avg_mid = sum([(tick.bid + tick.ask) / 2 for tick in recent_ticks]) / len(recent_ticks)
            spread_bps = float(avg_spread / avg_mid * 10000)
            
            # Market depth estimation (mock - in production use order book data)
            market_depth = self._estimate_market_depth(recent_ticks)
            
            # Liquidity score calculation
            liquidity_score = self._calculate_liquidity_score(recent_ticks, avg_spread)
            
            # Slippage estimation
            slippage_estimate = self._estimate_slippage(recent_ticks, avg_spread)
            
            # Create liquidity metrics
            metrics = LiquidityMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                bid_ask_spread=avg_spread,
                spread_bps=spread_bps,
                market_depth=market_depth,
                liquidity_score=liquidity_score,
                slippage_estimate=slippage_estimate
            )
            
            # Store metrics
            self.liquidity_cache[symbol] = metrics
            await self._store_liquidity_metrics(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity for {symbol}: {str(e)}")
    
    def _estimate_market_depth(self, ticks: List[TickData]) -> Dict[str, int]:
        """Estimate market depth from tick data"""
        try:
            # Mock implementation - in production use level 2 data
            total_volume = sum(tick.volume for tick in ticks)
            
            return {
                'bid_depth_level1': int(total_volume * 0.3),
                'ask_depth_level1': int(total_volume * 0.3),
                'bid_depth_level5': int(total_volume * 0.8),
                'ask_depth_level5': int(total_volume * 0.8)
            }
            
        except Exception as e:
            logger.error(f"Error estimating market depth: {str(e)}")
            return {}
    
    def _calculate_liquidity_score(self, ticks: List[TickData], avg_spread: Decimal) -> float:
        """Calculate overall liquidity score"""
        try:
            # Factors: spread tightness, volume consistency, tick frequency
            spread_score = max(0, 1 - float(avg_spread) * 10000)  # Lower spread = higher score
            
            volumes = [tick.volume for tick in ticks]
            volume_consistency = 1 - (np.std(volumes) / np.mean(volumes)) if volumes else 0
            
            tick_frequency = len(ticks) / 60  # Ticks per minute
            frequency_score = min(tick_frequency / 100, 1.0)  # Normalize to 100 ticks/min
            
            # Weighted combination
            liquidity_score = (spread_score * 0.4 + volume_consistency * 0.3 + frequency_score * 0.3)
            
            return max(0, min(liquidity_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.0
    
    def _estimate_slippage(self, ticks: List[TickData], avg_spread: Decimal) -> float:
        """Estimate execution slippage"""
        try:
            # Simple slippage model based on spread and volatility
            base_slippage = float(avg_spread) / 2  # Half spread as base
            
            # Add volatility component
            prices = [float(tick.last) for tick in ticks]
            volatility = np.std(prices) / np.mean(prices) if prices else 0
            
            volatility_component = volatility * 100  # Scale volatility
            
            return base_slippage + volatility_component
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {str(e)}")
            return 0.0
    
    async def get_order_flow_metrics(self, symbol: str) -> Optional[OrderFlowMetrics]:
        """Get current order flow metrics for symbol"""
        return self.order_flow_cache.get(symbol)
    
    async def get_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get current liquidity metrics for symbol"""
        return self.liquidity_cache.get(symbol)
    
    async def generate_microstructure_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate trading signals based on microstructure analysis"""
        try:
            signals = []
            
            order_flow = await self.get_order_flow_metrics(symbol)
            liquidity = await self.get_liquidity_metrics(symbol)
            
            if not order_flow or not liquidity:
                return signals
            
            # Order flow imbalance signals
            if abs(order_flow.imbalance_ratio) > 0.3:
                signal_type = "buy" if order_flow.imbalance_ratio > 0 else "sell"
                signals.append({
                    "type": signal_type,
                    "reason": "order_flow_imbalance",
                    "strength": abs(order_flow.imbalance_ratio),
                    "confidence": min(0.6 + abs(order_flow.imbalance_ratio) * 0.4, 0.9)
                })
            
            # Liquidity-based signals
            if liquidity.liquidity_score < 0.3:
                signals.append({
                    "type": "caution",
                    "reason": "low_liquidity",
                    "strength": 1 - liquidity.liquidity_score,
                    "confidence": 0.8
                })
            
            # Market impact signals
            if order_flow.market_impact > 0.01:  # 1% impact threshold
                signals.append({
                    "type": "caution",
                    "reason": "high_market_impact",
                    "strength": order_flow.market_impact,
                    "confidence": 0.7
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating microstructure signals for {symbol}: {str(e)}")
            return []
    
    async def _store_tick_analysis(self, symbol: str, analysis: Dict[str, Any]):
        """Store tick analysis results"""
        try:
            # Store in Redis with expiration
            key = f"tick_analysis:{symbol}"
            await self.redis_client.setex(key, 3600, str(analysis))  # 1 hour expiration
            
        except Exception as e:
            logger.error(f"Error storing tick analysis: {str(e)}")
    
    async def _store_order_flow_metrics(self, symbol: str, metrics: OrderFlowMetrics):
        """Store order flow metrics"""
        try:
            # Store in Redis
            key = f"order_flow:{symbol}"
            data = {
                'timestamp': metrics.timestamp.isoformat(),
                'buy_volume': metrics.buy_volume,
                'sell_volume': metrics.sell_volume,
                'imbalance_ratio': metrics.imbalance_ratio,
                'vwap': str(metrics.vwap),
                'twap': str(metrics.twap),
                'participation_rate': metrics.participation_rate,
                'market_impact': metrics.market_impact
            }
            await self.redis_client.setex(key, 1800, str(data))  # 30 minutes expiration
            
        except Exception as e:
            logger.error(f"Error storing order flow metrics: {str(e)}")
    
    async def _store_liquidity_metrics(self, symbol: str, metrics: LiquidityMetrics):
        """Store liquidity metrics"""
        try:
            # Store in Redis
            key = f"liquidity:{symbol}"
            data = {
                'timestamp': metrics.timestamp.isoformat(),
                'bid_ask_spread': str(metrics.bid_ask_spread),
                'spread_bps': metrics.spread_bps,
                'market_depth': metrics.market_depth,
                'liquidity_score': metrics.liquidity_score,
                'slippage_estimate': metrics.slippage_estimate
            }
            await self.redis_client.setex(key, 1800, str(data))  # 30 minutes expiration
            
        except Exception as e:
            logger.error(f"Error storing liquidity metrics: {str(e)}")
    
    async def _update_realtime_metrics(self, tick: TickData):
        """Update real-time metrics for dashboard"""
        try:
            # Calculate real-time spread
            spread = tick.ask - tick.bid
            spread_bps = float(spread / ((tick.bid + tick.ask) / 2) * 10000)
            
            # Store for real-time dashboard
            realtime_data = {
                'timestamp': tick.timestamp.isoformat(),
                'bid': str(tick.bid),
                'ask': str(tick.ask),
                'last': str(tick.last),
                'spread': str(spread),
                'spread_bps': spread_bps,
                'volume': tick.volume,
                'side': tick.side
            }
            
            await self.redis_client.setex(f"realtime:{tick.symbol}", 60, str(realtime_data))
            
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {str(e)}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                self.redis_client.close()
                await self.redis_client.wait_closed()
            
            logger.info("Market Microstructure Module cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
