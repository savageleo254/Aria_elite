"""
Strict Pydantic schemas for Market Microstructure API responses
ARIA-DAN Institutional Compliance - No empty schemas allowed
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

class OrderFlowResponse(BaseModel):
    """Order flow analysis response schema"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timestamp: str = Field(..., description="ISO timestamp")
    buy_volume: int = Field(..., ge=0, description="Buy volume in units")
    sell_volume: int = Field(..., ge=0, description="Sell volume in units")
    imbalance_ratio: float = Field(..., ge=-1, le=1, description="Order imbalance ratio")
    vwap: str = Field(..., description="Volume weighted average price")
    twap: str = Field(..., description="Time weighted average price")
    participation_rate: float = Field(..., ge=0, le=1, description="Market participation rate")
    market_impact: float = Field(..., ge=0, description="Estimated market impact")

class LiquidityResponse(BaseModel):
    """Liquidity analysis response schema"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: str = Field(..., description="ISO timestamp")
    bid_ask_spread: str = Field(..., description="Current bid-ask spread")
    spread_bps: float = Field(..., ge=0, description="Spread in basis points")
    market_depth: Dict[str, int] = Field(..., description="Market depth levels")
    liquidity_score: float = Field(..., ge=0, le=1, description="Liquidity quality score")
    slippage_estimate: float = Field(..., ge=0, description="Expected slippage")

class MarketImpactResponse(BaseModel):
    """Market impact calculation response schema"""
    symbol: str = Field(..., description="Trading symbol")
    order_size: int = Field(..., gt=0, description="Order size in units")
    side: str = Field(..., regex="^(buy|sell)$", description="Order side")
    estimated_impact_pips: float = Field(..., ge=0, description="Impact in pips")
    estimated_impact_percentage: float = Field(..., ge=0, description="Impact percentage")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    timestamp: str = Field(..., description="ISO timestamp")

class TickData(BaseModel):
    """Individual tick data schema"""
    timestamp: str = Field(..., description="ISO timestamp")
    bid: float = Field(..., gt=0, description="Bid price")
    ask: float = Field(..., gt=0, description="Ask price")
    last: float = Field(..., gt=0, description="Last trade price")
    volume: int = Field(..., gt=0, description="Tick volume")
    side: str = Field(..., regex="^(buy|sell)$", description="Trade side")

class LiveTicksResponse(BaseModel):
    """Live ticks response schema"""
    symbol: str = Field(..., description="Trading symbol")
    tick_count: int = Field(..., ge=0, description="Number of ticks returned")
    latest_timestamp: Optional[str] = Field(None, description="Latest tick timestamp")
    ticks: List[TickData] = Field(..., description="Array of tick data")

class TradingSignal(BaseModel):
    """Individual trading signal schema"""
    type: str = Field(..., regex="^(buy|sell|hold)$", description="Signal type")
    reason: str = Field(..., description="Signal generation reason")
    strength: float = Field(..., ge=0, le=1, description="Signal strength")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    timestamp: str = Field(..., description="ISO timestamp")

class MicrostructureSignalsResponse(BaseModel):
    """Microstructure signals response schema"""
    symbol: str = Field(..., description="Trading symbol")
    signal_count: int = Field(..., ge=0, description="Number of signals")
    timestamp: str = Field(..., description="ISO timestamp")
    signals: List[TradingSignal] = Field(..., description="Array of trading signals")

class TickAnalysisResponse(BaseModel):
    """Tick analysis response schema"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: str = Field(..., description="ISO timestamp")
    tick_count_1min: int = Field(..., ge=0, description="Tick count in 1 minute")
    tick_count_5min: int = Field(..., ge=0, description="Tick count in 5 minutes")
    price_clustering_score: float = Field(..., ge=0, le=1, description="Price clustering score")
    tick_imbalance: float = Field(..., ge=-1, le=1, description="Tick imbalance ratio")
    average_tick_size: float = Field(..., gt=0, description="Average tick size")
    tick_frequency: float = Field(..., ge=0, description="Tick frequency per second")
    momentum_persistence: float = Field(..., ge=0, le=1, description="Momentum persistence")

class ExecutionQualityResponse(BaseModel):
    """Execution quality metrics response schema"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: str = Field(..., description="ISO timestamp")
    average_slippage: float = Field(..., ge=0, description="Average slippage in pips")
    fill_rate: float = Field(..., ge=0, le=1, description="Order fill rate")
    reject_rate: float = Field(..., ge=0, le=1, description="Order rejection rate")
    execution_latency_ms: float = Field(..., ge=0, description="Execution latency in milliseconds")
    price_improvement_rate: float = Field(..., ge=0, le=1, description="Price improvement rate")
    effective_spread: float = Field(..., ge=0, description="Effective spread in pips")
    implementation_shortfall: float = Field(..., ge=0, description="Implementation shortfall")

class MarketProfileResponse(BaseModel):
    """Market profile analysis response schema"""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: str = Field(..., description="ISO timestamp")
    value_area_high: float = Field(..., gt=0, description="Value area high price")
    value_area_low: float = Field(..., gt=0, description="Value area low price")
    point_of_control: float = Field(..., gt=0, description="Point of control price")
    volume_profile: Dict[str, int] = Field(..., description="Volume at price levels")
    market_structure: str = Field(..., regex="^(balanced|trending|ranging)$", description="Market structure")
    distribution_type: str = Field(..., regex="^(normal|skewed|bimodal)$", description="Distribution type")

class HealthResponse(BaseModel):
    """System health response schema"""
    status: str = Field(..., regex="^(operational|degraded|error)$", description="System status")
    initialized: bool = Field(..., description="System initialization status")
    timestamp: str = Field(..., description="ISO timestamp")
    active_symbols: int = Field(..., ge=0, description="Number of active symbols")
    redis_connected: bool = Field(..., description="Redis connection status")

class ErrorResponse(BaseModel):
    """Standard error response schema"""
    detail: str = Field(..., description="Error detail message")
    error_code: Optional[str] = Field(None, description="Application error code")
    timestamp: str = Field(..., description="Error timestamp")
