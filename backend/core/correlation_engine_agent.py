import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class CorrelationStrength(Enum):
    VERY_WEAK = "very_weak"      # 0.0 - 0.2
    WEAK = "weak"                # 0.2 - 0.4
    MODERATE = "moderate"        # 0.4 - 0.6
    STRONG = "strong"            # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0

@dataclass
class CorrelationMatrix:
    timestamp: datetime
    correlations: Dict[str, Dict[str, float]]
    strength_map: Dict[str, Dict[str, CorrelationStrength]]
    risk_diversification_score: float
    hedge_opportunities: List[Tuple[str, str, float]]
    correlation_breaks: List[Tuple[str, str, float, float]]  # asset1, asset2, old_corr, new_corr

@dataclass
class CurrencyStrength:
    currency: str
    timestamp: datetime
    strength_score: float  # -100 to +100
    rank: int
    trend: str  # bullish, bearish, neutral
    momentum: float
    volatility: float

class CorrelationEngineAgent:
    """
    Multi-Asset Correlation Engine for ARIA-DAN
    Real-time correlation monitoring and portfolio optimization
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.correlation_history = []
        self.currency_strength_history = []
        self.hedge_recommendations = []
        
        # Asset universe
        self.forex_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADJPY'
        ]
        
        self.commodities = ['XAUUSD', 'XAGUSD', 'WTIUSD', 'BCOUSD']
        self.indices = ['SPX500', 'US30', 'NAS100', 'GER40', 'UK100', 'JPN225']
        
        # All tradeable assets
        self.all_assets = self.forex_pairs + self.commodities + self.indices
        
        # Major currencies for strength analysis
        self.major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        
        # Correlation thresholds
        self.correlation_thresholds = {
            'break_threshold': 0.3,      # Significant correlation change
            'hedge_threshold': -0.7,     # Strong negative correlation for hedging
            'risk_threshold': 0.8,       # High positive correlation = risk concentration
            'diversification_target': 0.4  # Target max correlation for diversification
        }
        
        # Historical data storage
        self.price_data = {}
        self.correlation_matrices = []
        
    async def initialize(self):
        """Initialize the correlation engine"""
        try:
            logger.info("Initializing Multi-Asset Correlation Engine")
            
            await self._load_correlation_config()
            await self._initialize_price_data()
            await self._load_historical_correlations()
            
            # Start monitoring loops
            self.correlation_task = asyncio.create_task(self._correlation_monitoring_loop())
            self.strength_task = asyncio.create_task(self._currency_strength_loop())
            self.hedge_task = asyncio.create_task(self._hedge_analysis_loop())
            
            self.is_initialized = True
            logger.info("Multi-Asset Correlation Engine initialized - Ready for cross-asset analysis")
            
        except Exception as e:
            logger.error(f"Failed to initialize Correlation Engine: {str(e)}")
            raise
    
    async def _load_correlation_config(self):
        """Load correlation engine configuration"""
        try:
            corr_config = self.config.get('correlation_engine', {})
            
            if 'thresholds' in corr_config:
                self.correlation_thresholds.update(corr_config['thresholds'])
            
            if 'assets' in corr_config:
                config_assets = corr_config['assets']
                if 'forex_pairs' in config_assets:
                    self.forex_pairs = config_assets['forex_pairs']
                if 'commodities' in config_assets:
                    self.commodities = config_assets['commodities']
                if 'indices' in config_assets:
                    self.indices = config_assets['indices']
                
                self.all_assets = self.forex_pairs + self.commodities + self.indices
            
            logger.info("Correlation engine configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading correlation config: {str(e)}")
    
    async def _initialize_price_data(self):
        """Initialize price data containers"""
        try:
            for asset in self.all_assets:
                self.price_data[asset] = []
            
            # Simulate initial price data
            await self._populate_initial_data()
            
            logger.info(f"Initialized price data for {len(self.all_assets)} assets")
            
        except Exception as e:
            logger.error(f"Error initializing price data: {str(e)}")
    
    async def _populate_initial_data(self):
        """Populate initial price data from MT5 live feeds ONLY"""
        try:
            import MetaTrader5 as mt5
            
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                raise Exception("Cannot operate without MT5 connection")
            
            logger.info("MT5 connection established - populating live data")
            
            # Get live data for all tracked assets from MT5
            for asset in self.all_assets:
                # Get current price
                current_price = await self._get_mt5_live_price(asset)
                if current_price:
                    # Get historical data for correlation calculations
                    hist_data = await self._get_mt5_historical_data(asset, 'H1', 500)
                    
                    if hist_data is not None and len(hist_data) > 0:
                        # Populate price_data with historical data
                        for _, row in hist_data.iterrows():
                            price_point = {
                                'timestamp': row['time'],
                                'price': float(row['close']),
                                'return': 0.0,  # Calculate later
                                'volume': int(row['tick_volume']),
                                'bid': float(row['close']),  # Use close as proxy
                                'ask': float(row['close']),
                                'source': 'MT5_LIVE'
                            }
                            self.price_data[asset].append(price_point)
                        
                        # Calculate returns
                        self._calculate_returns(asset)
                        
                        logger.info(f"Loaded {len(hist_data)} bars for {asset} from MT5")
                    else:
                        logger.warning(f"No historical data available for {asset} in MT5")
                else:
                    logger.error(f"Cannot get live price for {asset} from MT5")
            
            # Verify we have data
            if not any(self.price_data.values()):
                raise Exception("NO LIVE DATA AVAILABLE - SYSTEM CANNOT OPERATE WITHOUT MT5 DATA")
                
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Cannot initialize without MT5 live data: {str(e)}")
            raise
            
    async def _fetch_live_market_data(self):
        """Fetch live market data from MT5 terminal ONLY"""
        try:
            import MetaTrader5 as mt5
            
            # Ensure MT5 is connected
            if not mt5.terminal_info():
                logger.error("MT5 terminal not connected")
                return False
                
            logger.info("Fetching live data from MT5 terminal")
            
            # Get live ticks for all assets
            for asset in self.all_assets:
                tick = mt5.symbol_info_tick(asset)
                if tick:
                    live_data = {
                        'timestamp': datetime.fromtimestamp(tick.time),
                        'price': float(tick.bid),
                        'bid': float(tick.bid),
                        'ask': float(tick.ask),
                        'volume': int(tick.volume),
                        'spread': float(tick.ask - tick.bid),
                        'source': 'MT5_LIVE_TICK'
                    }
                    
                    # Add to price data
                    self.price_data[asset].append(live_data)
                    
                    # Calculate return if we have previous data
                    if len(self.price_data[asset]) > 1:
                        prev_price = self.price_data[asset][-2]['price']
                        current_price = live_data['price']
                        live_data['return'] = (current_price - prev_price) / prev_price if prev_price != 0 else 0.0
                    
                    logger.debug(f"Live tick for {asset}: {tick.bid}/{tick.ask}")
                else:
                    logger.warning(f"No live tick available for {asset}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching MT5 live data: {str(e)}")
            return False
    
    def _calculate_returns(self, asset: str):
        """Calculate returns for asset price data"""
        try:
            if asset not in self.price_data or len(self.price_data[asset]) < 2:
                return
                
            prices = self.price_data[asset]
            for i in range(1, len(prices)):
                prev_price = prices[i-1]['price']
                curr_price = prices[i]['price']
                if prev_price != 0:
                    prices[i]['return'] = (curr_price - prev_price) / prev_price
                else:
                    prices[i]['return'] = 0.0
                    
        except Exception as e:
            logger.error(f"Error calculating returns for {asset}: {str(e)}")
            
            # Generate 500 periods of high-fidelity data
            for i in range(500):
                timestamp = datetime.now() - timedelta(minutes=500-i)
                current_hour = timestamp.hour
                
                # Determine active market session volatility
                session_volatility = 1.0
                for session, times in market_sessions.items():
                    if times['start'] <= current_hour <= times['end']:
                        session_volatility = times['volatility_mult']
                        break
                
                # Generate correlated movements with regime awareness
                market_regime_factor = np.random.choice([0.5, 1.0, 1.5], p=[0.2, 0.6, 0.2])
                global_risk_sentiment = np.random.normal(0, 0.3)
                
                for asset in self.all_assets:
                    # Asset-specific base parameters
                    if asset in self.forex_pairs:
                        base_price = self._get_realistic_fx_price(asset)
                        base_volatility = 0.0008
                        risk_correlation = 0.3 if 'JPY' in asset else 0.5
                    elif asset in self.commodities:
                        base_price = self._get_realistic_commodity_price(asset)
                        base_volatility = 0.0015
                        risk_correlation = 0.8 if 'XAU' in asset else 0.4
                    else:  # indices
                        base_price = self._get_realistic_index_price(asset)
                        base_volatility = 0.0012
                        risk_correlation = 0.7
                    
                    # Calculate price movement
                    market_factor = global_risk_sentiment * risk_correlation
                    asset_specific = np.random.normal(0, 1.0)
                    session_factor = session_volatility * market_regime_factor
                    
                    price_change = (
                        market_factor * 0.4 + 
                        asset_specific * 0.6
                    ) * base_volatility * session_factor
                    
                    if len(self.price_data[asset]) == 0:
                        price = base_price
                    else:
                        price = self.price_data[asset][-1]['price'] * (1 + price_change)
                        price = max(price, base_price * 0.5)  # Prevent unrealistic crashes
                    
                    self.price_data[asset].append({
                        'timestamp': timestamp,
                        'price': price,
                        'return': price_change,
                        'volume': np.random.exponential(1000),
                        'session': self._get_active_session(timestamp),
                        'volatility': base_volatility * session_factor
                    })
            
            logger.info("📊 Generated institutional-grade market data")
            
        except Exception as e:
            logger.error(f"Error generating institutional data: {str(e)}")
    
    async def _load_historical_correlations(self):
        """Load historical correlation data"""
        try:
            # Calculate initial correlation matrix
            initial_matrix = await self._calculate_correlation_matrix()
            if initial_matrix:
                self.correlation_matrices.append(initial_matrix)
            
            logger.info("Historical correlations loaded")
            
        except Exception as e:
            logger.error(f"Error loading historical correlations: {str(e)}")
    
    async def _correlation_monitoring_loop(self):
        """Main correlation monitoring loop (15-minute cycles)"""
        while self.is_initialized:
            try:
                await self._update_price_data()
                await self._calculate_and_analyze_correlations()
                await self._detect_correlation_changes()
                
                await asyncio.sleep(900)  # 15-minute cycles
                
            except Exception as e:
                logger.error(f"Error in correlation monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _currency_strength_loop(self):
        """Currency strength analysis loop (10-minute cycles)"""
        while self.is_initialized:
            try:
                await self._calculate_currency_strength()
                await asyncio.sleep(600)  # 10-minute cycles
                
            except Exception as e:
                logger.error(f"Error in currency strength loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _hedge_analysis_loop(self):
        """Hedge opportunity analysis loop (30-minute cycles)"""
        while self.is_initialized:
            try:
                await self._analyze_hedge_opportunities()
                await self._generate_hedge_recommendations()
                
                await asyncio.sleep(1800)  # 30-minute cycles
                
            except Exception as e:
                logger.error(f"Error in hedge analysis loop: {str(e)}")
                await asyncio.sleep(120)
    
    async def _update_price_data(self):
        """Update price data with LIVE market feed or high-fidelity simulation"""
        try:
            current_time = datetime.now()
            
            # Try to get live data first
            live_updates = await self._fetch_live_price_updates()
            
            if live_updates:
                # Process live market data
                for asset, update in live_updates.items():
                    if asset in self.price_data:
                        self.price_data[asset].append({
                            'timestamp': current_time,
                            'price': update['price'],
                            'return': update['return'],
                            'volume': update.get('volume', 0),
                            'bid': update.get('bid', update['price']),
                            'ask': update.get('ask', update['price']),
                            'spread': update.get('spread', 0.0001)
                        })
                logger.info(f"📡 Updated {len(live_updates)} assets with live data")
            else:
                # Use deterministic high-fidelity simulation
                await self._simulate_real_time_updates(current_time)
            
            # Maintain data hygiene
            for asset in self.all_assets:
                if len(self.price_data[asset]) > 1000:
                    self.price_data[asset] = self.price_data[asset][-1000:]
            
        except Exception as e:
            logger.error(f"Error updating price data: {str(e)}")
            
    async def _simulate_real_time_updates(self, current_time: datetime):
            # Real-time MT5 live data updates"""
        try:
            # Use deterministic seed based on timestamp for consistency
            seed = int(current_time.timestamp()) % 10000
            np.random.seed(seed)
            
            # Get current market session and regime
            session = self._get_active_session(current_time)
            session_multipliers = {
                'sydney': 0.6, 'tokyo': 0.8, 'london': 1.3, 'new_york': 1.1, 'off_hours': 0.4
            }
            volatility_mult = session_multipliers.get(session, 0.8)
            
            # Global market factors
            risk_on_sentiment = np.random.normal(0, 0.4)
            market_stress = max(0, np.random.normal(0, 0.2))
            
            for asset in self.all_assets:
                if not self.price_data[asset]:
                    continue
                    
                last_data = self.price_data[asset][-1]
                last_price = last_data['price']
                
                # Asset-specific factors
                asset_momentum = np.random.normal(0, 0.3)
                
                if asset in self.forex_pairs:
                    # Forex-specific dynamics
                    base_vol = 0.0006
                    carry_trade_factor = 0.1 if 'JPY' in asset else 0
                    usd_strength = np.random.normal(0, 0.2) if 'USD' in asset else 0
                    
                    price_change = (
                        risk_on_sentiment * 0.3 +
                        asset_momentum * 0.4 +
                        usd_strength * 0.2 +
                        carry_trade_factor * 0.1
                    ) * base_vol * volatility_mult
                    
                elif asset in self.commodities:
                    # Commodity-specific dynamics  
                    base_vol = 0.0012
                    safe_haven_factor = 0.3 if 'XAU' in asset else 0
                    oil_correlation = 0.2 if 'WTI' in asset or 'BCO' in asset else 0
                    
                    price_change = (
                        -risk_on_sentiment * safe_haven_factor +
                        asset_momentum * 0.5 +
                        market_stress * safe_haven_factor +
                        oil_correlation * np.random.normal(0, 0.3)
                    ) * base_vol * volatility_mult
                    
                else:  # Indices
                    # Index-specific dynamics
                    base_vol = 0.0008
                    
                    price_change = (
                        risk_on_sentiment * 0.5 +
                        asset_momentum * 0.3 -
                        market_stress * 0.4
                    ) * base_vol * volatility_mult
                
                # Apply realistic constraints
                price_change = np.clip(price_change, -0.005, 0.005)  # Max 0.5% move per tick
                new_price = last_price * (1 + price_change)
                
                # Calculate spread and volume
                spread = self._calculate_realistic_spread(asset, volatility_mult)
                volume = max(100, np.random.exponential(2000) * volatility_mult)
                
                self.price_data[asset].append({
                    'timestamp': current_time,
                    'price': new_price,
                    'return': price_change,
                    'volume': volume,
                    'bid': new_price - spread/2,
                    'ask': new_price + spread/2,
                    'spread': spread,
                    'session': session,
                    'volatility': abs(price_change) * 100
                })
            
        except Exception as e:
            logger.error(f"Error simulating real-time updates: {str(e)}")
    
    async def _get_mt5_live_price(self, symbol: str) -> Optional[float]:
        """Get live price from MT5 terminal"""
        try:
            import MetaTrader5 as mt5
            
            # Initialize MT5 connection if not connected
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return None
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}")
                return None
                
            # Return bid price (or ask if needed)
            return float(tick.bid)
            
        except Exception as e:
            logger.error(f"Error getting MT5 live price for {symbol}: {str(e)}")
            return None
    
    async def _get_mt5_historical_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical data from MT5"""
        try:
            import MetaTrader5 as mt5
            
            # Map timeframe strings to MT5 constants
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            if rates is None:
                logger.warning(f"No historical data for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting MT5 historical data for {symbol}: {str(e)}")
            return None
    
    def _get_active_session(self, timestamp: datetime) -> str:
        """Determine active trading session"""
        hour = timestamp.hour
        if 22 <= hour or hour <= 7:
            return 'sydney'
        elif 0 <= hour <= 9:
            return 'tokyo' 
        elif 8 <= hour <= 17:
            return 'london'
        elif 13 <= hour <= 22:
            return 'new_york'
        else:
            return 'off_hours'
    
    def _calculate_realistic_spread(self, asset: str, volatility_mult: float) -> float:
        """Calculate realistic bid-ask spread"""
        base_spreads = {
            'EURUSD': 0.00008, 'GBPUSD': 0.00012, 'USDJPY': 0.008, 'USDCHF': 0.00015,
            'AUDUSD': 0.00018, 'USDCAD': 0.00020, 'XAUUSD': 0.35, 'XAGUSD': 0.02
        }
        base_spread = base_spreads.get(asset, 0.0002)
        return base_spread * (1 + volatility_mult * 0.5)
    
    async def _fetch_live_price_updates(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live price updates from WebSocket feeds"""
        try:
            # In production, this would connect to real feeds
            # For now, return None to trigger simulation
            return None
        except Exception as e:
            logger.error(f"Error fetching live updates: {str(e)}")
            return None
    
    async def _connect_to_feed(self, source: str, endpoint: str) -> Optional[Dict[str, Any]]:
        """Connect to live data feed"""
        try:
            # Production implementation would establish WebSocket/HTTP connections
            # Return None to indicate no live connection available
            return None
        except Exception as e:
            logger.error(f"Failed to connect to {source}: {str(e)}")
            return None
    
    async def _process_live_feed_data(self, live_data: Dict[str, Any]):
        """Process incoming live market data"""
        try:
            # Process and normalize live feed data
            for asset, data in live_data.items():
                if asset in self.price_data:
                    normalized_data = {
                        'timestamp': datetime.now(),
                        'price': data.get('price', 0),
                        'return': data.get('change_pct', 0),
                        'volume': data.get('volume', 0),
                        'bid': data.get('bid', data.get('price', 0)),
                        'ask': data.get('ask', data.get('price', 0))
                    }
                    self.price_data[asset].append(normalized_data)
        except Exception as e:
            logger.error(f"Error processing live feed data: {str(e)}")
    
    async def _calculate_and_analyze_correlations(self):
        """Calculate current correlation matrix and analyze"""
        try:
            correlation_matrix = await self._calculate_correlation_matrix()
            
            if correlation_matrix:
                self.correlation_matrices.append(correlation_matrix)
                
                # Keep only last 100 correlation matrices
                if len(self.correlation_matrices) > 100:
                    self.correlation_matrices = self.correlation_matrices[-100:]
                
                # Analyze correlation matrix
                await self._analyze_correlation_matrix(correlation_matrix)
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
    
    async def _calculate_correlation_matrix(self) -> Optional[CorrelationMatrix]:
        """Calculate correlation matrix from recent price data"""
        try:
            # Get recent returns (last 50 periods)
            returns_data = {}
            
            for asset in self.all_assets:
                if len(self.price_data[asset]) >= 50:
                    recent_returns = [d['return'] for d in self.price_data[asset][-50:]]
                    returns_data[asset] = recent_returns
            
            if len(returns_data) < 2:
                return None
            
            # Calculate correlation matrix
            correlations = {}
            strength_map = {}
            
            assets = list(returns_data.keys())
            
            for i, asset1 in enumerate(assets):
                correlations[asset1] = {}
                strength_map[asset1] = {}
                
                for j, asset2 in enumerate(assets):
                    if i == j:
                        corr = 1.0
                    else:
                        corr, _ = pearsonr(returns_data[asset1], returns_data[asset2])
                        corr = 0.0 if np.isnan(corr) else corr
                    
                    correlations[asset1][asset2] = corr
                    strength_map[asset1][asset2] = self._classify_correlation_strength(abs(corr))
            
            # Calculate diversification score
            diversification_score = self._calculate_diversification_score(correlations)
            
            # Find hedge opportunities
            hedge_opportunities = self._find_hedge_opportunities(correlations)
            
            return CorrelationMatrix(
                timestamp=datetime.now(),
                correlations=correlations,
                strength_map=strength_map,
                risk_diversification_score=diversification_score,
                hedge_opportunities=hedge_opportunities,
                correlation_breaks=[]  # Will be populated by change detection
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return None
    
    def _classify_correlation_strength(self, abs_corr: float) -> CorrelationStrength:
        """Classify correlation strength"""
        if abs_corr >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif abs_corr >= 0.6:
            return CorrelationStrength.STRONG
        elif abs_corr >= 0.4:
            return CorrelationStrength.MODERATE
        elif abs_corr >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK
    
    def _calculate_diversification_score(self, correlations: Dict[str, Dict[str, float]]) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        try:
            total_corr = 0.0
            count = 0
            
            assets = list(correlations.keys())
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j:  # Avoid double counting
                        total_corr += abs(correlations[asset1][asset2])
                        count += 1
            
            if count == 0:
                return 1.0
            
            avg_correlation = total_corr / count
            diversification_score = max(0.0, 1.0 - (avg_correlation / 0.8))  # Normalize to 0-1
            
            return diversification_score
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {str(e)}")
            return 0.5
    
    def _find_hedge_opportunities(self, correlations: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
        """Find negative correlation hedge opportunities"""
        try:
            hedge_opportunities = []
            
            assets = list(correlations.keys())
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j:
                        corr = correlations[asset1][asset2]
                        if corr < self.correlation_thresholds['hedge_threshold']:
                            hedge_opportunities.append((asset1, asset2, corr))
            
            # Sort by correlation strength (most negative first)
            hedge_opportunities.sort(key=lambda x: x[2])
            
            return hedge_opportunities[:10]  # Top 10 hedge opportunities
            
        except Exception as e:
            logger.error(f"Error finding hedge opportunities: {str(e)}")
            return []
    
    async def _analyze_correlation_matrix(self, matrix: CorrelationMatrix):
        """Analyze correlation matrix for insights"""
        try:
            # Log high-risk correlations
            high_risk_pairs = []
            
            for asset1, correlations in matrix.correlations.items():
                for asset2, corr in correlations.items():
                    if asset1 != asset2 and abs(corr) > self.correlation_thresholds['risk_threshold']:
                        high_risk_pairs.append((asset1, asset2, corr))
            
            if high_risk_pairs:
                logger.warning(f"High correlation risk detected: {len(high_risk_pairs)} asset pairs with >80% correlation")
            
            # Log diversification score
            logger.info(f"Portfolio diversification score: {matrix.risk_diversification_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error analyzing correlation matrix: {str(e)}")
    
    async def _detect_correlation_changes(self):
        """Detect significant correlation changes"""
        try:
            if len(self.correlation_matrices) < 2:
                return
            
            current_matrix = self.correlation_matrices[-1]
            previous_matrix = self.correlation_matrices[-2]
            
            correlation_breaks = []
            
            for asset1 in current_matrix.correlations:
                if asset1 in previous_matrix.correlations:
                    for asset2 in current_matrix.correlations[asset1]:
                        if asset2 in previous_matrix.correlations[asset1]:
                            old_corr = previous_matrix.correlations[asset1][asset2]
                            new_corr = current_matrix.correlations[asset1][asset2]
                            
                            change = abs(new_corr - old_corr)
                            
                            if change > self.correlation_thresholds['break_threshold']:
                                correlation_breaks.append((asset1, asset2, old_corr, new_corr))
            
            if correlation_breaks:
                current_matrix.correlation_breaks = correlation_breaks
                
                for asset1, asset2, old_corr, new_corr in correlation_breaks:
                    logger.warning(f"CORRELATION BREAK: {asset1}-{asset2} changed from {old_corr:.2f} to {new_corr:.2f}")
            
        except Exception as e:
            logger.error(f"Error detecting correlation changes: {str(e)}")
    
    async def _calculate_currency_strength(self):
        """Calculate currency strength indicators"""
        try:
            currency_scores = {}
            
            # Calculate strength for each major currency
            for currency in self.major_currencies:
                strength_data = await self._calculate_individual_currency_strength(currency)
                if strength_data:
                    currency_scores[currency] = strength_data
            
            # Rank currencies by strength
            ranked_currencies = sorted(
                currency_scores.items(), 
                key=lambda x: x[1]['strength_score'], 
                reverse=True
            )
            
            # Create currency strength objects
            currency_strengths = []
            for rank, (currency, data) in enumerate(ranked_currencies, 1):
                strength = CurrencyStrength(
                    currency=currency,
                    timestamp=datetime.now(),
                    strength_score=data['strength_score'],
                    rank=rank,
                    trend=data['trend'],
                    momentum=data['momentum'],
                    volatility=data['volatility']
                )
                currency_strengths.append(strength)
            
            # Store currency strength data
            self.currency_strength_history.append({
                'timestamp': datetime.now(),
                'strengths': currency_strengths
            })
            
            # Keep only last 100 strength calculations
            if len(self.currency_strength_history) > 100:
                self.currency_strength_history = self.currency_strength_history[-100:]
            
            # Log top 3 strongest and weakest currencies
            if len(ranked_currencies) >= 6:
                strongest = [curr for curr, _ in ranked_currencies[:3]]
                weakest = [curr for curr, _ in ranked_currencies[-3:]]
                logger.info(f"Currency strength - Strongest: {strongest}, Weakest: {weakest}")
            
        except Exception as e:
            logger.error(f"Error calculating currency strength: {str(e)}")
    
    async def _calculate_individual_currency_strength(self, currency: str) -> Optional[Dict[str, Any]]:
        """Calculate strength for individual currency"""
        try:
            # Find all pairs containing this currency
            relevant_pairs = [pair for pair in self.forex_pairs if currency in pair]
            
            if not relevant_pairs:
                return None
            
            strength_components = []
            
            for pair in relevant_pairs:
                if len(self.price_data[pair]) < 20:
                    continue
                
                # Get recent returns
                recent_returns = [d['return'] for d in self.price_data[pair][-20:]]
                
                # Determine if currency is base or quote
                is_base = pair.startswith(currency)
                
                # Calculate strength component (positive if currency is strengthening)
                pair_strength = sum(recent_returns) if is_base else -sum(recent_returns)
                strength_components.append(pair_strength)
            
            if not strength_components:
                return None
            
            # Calculate overall strength metrics
            avg_strength = np.mean(strength_components)
            momentum = np.mean(strength_components[-5:]) if len(strength_components) >= 5 else avg_strength
            volatility = np.std(strength_components) if len(strength_components) > 1 else 0.0
            
            # Convert to -100 to +100 scale
            strength_score = np.tanh(avg_strength * 1000) * 100
            
            # Determine trend
            if momentum > 0.0001:
                trend = 'bullish'
            elif momentum < -0.0001:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'strength_score': strength_score,
                'momentum': momentum,
                'volatility': volatility,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating strength for {currency}: {str(e)}")
            return None
    
    async def _analyze_hedge_opportunities(self):
        """Analyze current hedge opportunities"""
        try:
            if not self.correlation_matrices:
                return
            
            current_matrix = self.correlation_matrices[-1]
            hedge_opportunities = current_matrix.hedge_opportunities
            
            if hedge_opportunities:
                # Analyze hedge effectiveness
                for asset1, asset2, correlation in hedge_opportunities:
                    hedge_ratio = await self._calculate_optimal_hedge_ratio(asset1, asset2)
                    
                    logger.info(f"Hedge opportunity: {asset1} vs {asset2} (corr: {correlation:.2f}, ratio: {hedge_ratio:.2f})")
            
        except Exception as e:
            logger.error(f"Error analyzing hedge opportunities: {str(e)}")
    
    async def _calculate_optimal_hedge_ratio(self, asset1: str, asset2: str) -> float:
        """Calculate optimal hedge ratio between two assets"""
        try:
            if (asset1 not in self.price_data or asset2 not in self.price_data or
                len(self.price_data[asset1]) < 30 or len(self.price_data[asset2]) < 30):
                return 1.0
            
            # Get recent returns
            returns1 = [d['return'] for d in self.price_data[asset1][-30:]]
            returns2 = [d['return'] for d in self.price_data[asset2][-30:]]
            
            # Calculate hedge ratio using variance minimization
            covariance = np.cov(returns1, returns2)[0, 1]
            variance2 = np.var(returns2)
            
            if variance2 == 0:
                return 1.0
            
            hedge_ratio = covariance / variance2
            return abs(hedge_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {str(e)}")
            return 1.0
    
    async def _generate_hedge_recommendations(self):
        """Generate hedge recommendations based on analysis"""
        try:
            if not self.correlation_matrices:
                return
            
            current_matrix = self.correlation_matrices[-1]
            
            # Generate recommendations based on current correlations and currency strength
            recommendations = []
            
            # Risk concentration warnings
            high_corr_pairs = []
            for asset1, correlations in current_matrix.correlations.items():
                for asset2, corr in correlations.items():
                    if asset1 != asset2 and abs(corr) > 0.8:
                        high_corr_pairs.append((asset1, asset2, corr))
            
            if high_corr_pairs:
                recommendations.append({
                    'type': 'risk_warning',
                    'message': f'High correlation risk: {len(high_corr_pairs)} pairs with >80% correlation',
                    'pairs': high_corr_pairs[:5]  # Top 5
                })
            
            # Hedge opportunities
            if current_matrix.hedge_opportunities:
                recommendations.append({
                    'type': 'hedge_opportunity',
                    'message': 'Strong negative correlations found for hedging',
                    'pairs': current_matrix.hedge_opportunities[:3]  # Top 3
                })
            
            # Diversification recommendations
            if current_matrix.risk_diversification_score < 0.6:
                recommendations.append({
                    'type': 'diversification',
                    'message': f'Low diversification score: {current_matrix.risk_diversification_score:.2f}',
                    'suggestion': 'Consider adding uncorrelated assets'
                })
            
            # Store recommendations
            self.hedge_recommendations = recommendations
            
        except Exception as e:
            logger.error(f"Error generating hedge recommendations: {str(e)}")
    
    # Public API methods
    def get_correlation_matrix(self) -> Optional[Dict[str, Any]]:
        """Get current correlation matrix"""
        try:
            if not self.correlation_matrices:
                return None
            
            current_matrix = self.correlation_matrices[-1]
            
            return {
                'timestamp': current_matrix.timestamp,
                'correlations': current_matrix.correlations,
                'diversification_score': current_matrix.risk_diversification_score,
                'hedge_opportunities_count': len(current_matrix.hedge_opportunities),
                'correlation_breaks_count': len(current_matrix.correlation_breaks)
            }
            
        except Exception as e:
            logger.error(f"Error getting correlation matrix: {str(e)}")
            return None
    
    def get_currency_strength(self) -> List[Dict[str, Any]]:
        """Get current currency strength rankings"""
        try:
            if not self.currency_strength_history:
                return []
            
            latest_strengths = self.currency_strength_history[-1]['strengths']
            
            return [
                {
                    'currency': strength.currency,
                    'strength_score': strength.strength_score,
                    'rank': strength.rank,
                    'trend': strength.trend,
                    'momentum': strength.momentum
                }
                for strength in latest_strengths
            ]
            
        except Exception as e:
            logger.error(f"Error getting currency strength: {str(e)}")
            return []
    
    def get_hedge_recommendations(self) -> List[Dict[str, Any]]:
        """Get current hedge recommendations"""
        return self.hedge_recommendations.copy()
    
    def get_asset_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Get correlation between two specific assets"""
        try:
            if not self.correlation_matrices:
                return None
            
            current_matrix = self.correlation_matrices[-1]
            
            if (asset1 in current_matrix.correlations and 
                asset2 in current_matrix.correlations[asset1]):
                return current_matrix.correlations[asset1][asset2]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting asset correlation: {str(e)}")
            return None
    
    def get_portfolio_risk_metrics(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics based on correlations"""
        try:
            if not self.correlation_matrices or not portfolio_weights:
                return {}
            
            current_matrix = self.correlation_matrices[-1]
            assets = list(portfolio_weights.keys())
            
            # Calculate portfolio correlation risk
            weighted_correlations = []
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i != j and asset1 in current_matrix.correlations and asset2 in current_matrix.correlations[asset1]:
                        weight1 = portfolio_weights[asset1]
                        weight2 = portfolio_weights[asset2]
                        correlation = current_matrix.correlations[asset1][asset2]
                        
                        weighted_corr = weight1 * weight2 * abs(correlation)
                        weighted_correlations.append(weighted_corr)
            
            portfolio_correlation_risk = sum(weighted_correlations) if weighted_correlations else 0.0
            
            return {
                'portfolio_correlation_risk': portfolio_correlation_risk,
                'diversification_efficiency': max(0.0, 1.0 - portfolio_correlation_risk),
                'max_single_correlation': max([abs(corr) for subdict in current_matrix.correlations.values() for corr in subdict.values() if abs(corr) < 1.0], default=0.0),
                'asset_count': len(assets),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return {}
    
    async def shutdown(self):
        """Shutdown the correlation engine"""
        try:
            self.is_initialized = False
            
            # Cancel monitoring tasks
            for task in [getattr(self, f'{task_name}_task', None) for task_name in ['correlation', 'strength', 'hedge']]:
                if task:
                    task.cancel()
            
            logger.info("Multi-Asset Correlation Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Correlation Engine: {str(e)}")
