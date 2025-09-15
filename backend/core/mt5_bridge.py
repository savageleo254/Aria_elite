import asyncio
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from collections import deque, defaultdict

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP" 
    SELL_STOP = "SELL_STOP"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"

@dataclass
class MT5Position:
    ticket: int
    symbol: str
    position_type: str
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    comment: str
    magic: int
    timestamp: datetime

@dataclass  
class MT5Order:
    ticket: int
    symbol: str
    order_type: str
    volume: float
    price: float
    sl: float
    tp: float
    magic: int
    comment: str
    timestamp: datetime

class MT5Bridge:
    """
    ARIA-DAN MT5 Bridge - Live Market Data & Execution Engine
    Production-grade MetaTrader 5 integration for institutional trading
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.is_connected = False
        
        # Connection settings
        self.account_info = None
        self.terminal_info = None
        
        # Live data streams
        self.live_prices = defaultdict(dict)
        self.price_stream = deque(maxlen=10000)
        self.tick_subscribers = []
        
        # Position management
        self.open_positions = {}
        self.order_history = deque(maxlen=5000)
        self.execution_queue = asyncio.Queue()
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2%
        self.max_daily_drawdown = 0.05  # 5%
        self.max_open_positions = 10
        
        # Magic numbers for ARIA system
        self.aria_magic_base = 777000
        self.magic_counter = 0
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'slippage_total': 0.0,
            'avg_execution_time': 0.0
        }
        
    async def initialize(self):
        """Initialize MT5 connection and data streams"""
        try:
            logger.info("Initializing ARIA-DAN MT5 Bridge")
            
            # Initialize MT5 terminal
            if not mt5.initialize():
                error = mt5.last_error()
                raise Exception(f"MT5 initialization failed: {error}")
            
            # Get account and terminal info
            await self._get_connection_info()
            
            # Validate account for live trading
            await self._validate_account()
            
            # Start data streams
            await self._start_live_data_streams()
            
            # Start execution engine
            await self._start_execution_engine()
            
            self.is_initialized = True
            self.is_connected = True
            
            logger.info(f"MT5 Bridge initialized - Account: {self.account_info.login}, Balance: ${self.account_info.balance:.2f}")
            
        except Exception as e:
            logger.error(f"MT5 Bridge initialization failed: {str(e)}")
            raise
            
    async def _get_connection_info(self):
        """Get MT5 account and terminal information"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                raise Exception("Failed to get account info")
            
            self.account_info = account_info
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                raise Exception("Failed to get terminal info")
                
            self.terminal_info = terminal_info
            
            logger.info(f"Account: {account_info.login}, Company: {account_info.company}")
            logger.info(f"Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
            
        except Exception as e:
            logger.error(f"Error getting connection info: {str(e)}")
            raise
    
    async def _validate_account(self):
        """Validate account is suitable for live trading"""
        try:
            if not self.account_info.trade_allowed:
                raise Exception("Trading is not allowed on this account")
                
            if self.account_info.trade_expert:
                logger.info("Expert Advisor trading enabled")
            else:
                logger.warning("Expert Advisor trading is DISABLED - manual intervention required")
            
            # Check minimum balance
            min_balance = 1000.0  # Minimum $1000 for live trading
            if self.account_info.balance < min_balance:
                logger.warning(f"Account balance ${self.account_info.balance:.2f} below recommended minimum ${min_balance}")
            
        except Exception as e:
            logger.error(f"Account validation failed: {str(e)}")
            raise
    
    async def _start_live_data_streams(self):
        """Start live price data streams"""
        try:
            logger.info("Starting live data streams")
            
            # Start price monitoring task
            asyncio.create_task(self._price_monitoring_loop())
            
            # Start position monitoring
            asyncio.create_task(self._position_monitoring_loop())
            
        except Exception as e:
            logger.error(f"Error starting data streams: {str(e)}")
            raise
    
    async def _start_execution_engine(self):
        """Start trade execution engine"""
        try:
            logger.info("Starting execution engine")
            
            # Start execution worker
            asyncio.create_task(self._execution_worker())
            
        except Exception as e:
            logger.error(f"Error starting execution engine: {str(e)}")
            raise
    
    async def _price_monitoring_loop(self):
        """Monitor live price data"""
        while self.is_connected:
            try:
                # Get symbols we're monitoring
                symbols = self._get_monitored_symbols()
                
                for symbol in symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        price_data = {
                            'symbol': symbol,
                            'bid': float(tick.bid),
                            'ask': float(tick.ask),
                            'last': float(tick.last),
                            'volume': int(tick.volume),
                            'time': datetime.fromtimestamp(tick.time),
                            'spread': float(tick.ask - tick.bid)
                        }
                        
                        # Update live prices
                        self.live_prices[symbol] = price_data
                        
                        # Add to price stream
                        self.price_stream.append(price_data)
                        
                        # Notify subscribers
                        await self._notify_tick_subscribers(symbol, price_data)
                
                await asyncio.sleep(0.1)  # 100ms monitoring
                
            except Exception as e:
                logger.error(f"Error in price monitoring: {str(e)}")
                await asyncio.sleep(1)
    
    async def _position_monitoring_loop(self):
        """Monitor open positions"""
        while self.is_connected:
            try:
                # Get current positions
                positions = mt5.positions_get()
                if positions:
                    current_positions = {}
                    
                    for pos in positions:
                        mt5_pos = MT5Position(
                            ticket=pos.ticket,
                            symbol=pos.symbol,
                            position_type="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                            volume=pos.volume,
                            price_open=pos.price_open,
                            price_current=pos.price_current,
                            profit=pos.profit,
                            swap=pos.swap,
                            commission=pos.commission,
                            comment=pos.comment,
                            magic=pos.magic,
                            timestamp=datetime.fromtimestamp(pos.time)
                        )
                        
                        current_positions[pos.ticket] = mt5_pos
                    
                    # Update positions
                    self.open_positions = current_positions
                
                await asyncio.sleep(1)  # 1 second position monitoring
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {str(e)}")
                # Continue monitoring even if there's an error
                await asyncio.sleep(5)
    
    async def _execution_worker(self):
        """Process trade execution queue"""
        while self.is_connected:
            try:
                # Get next execution request
                execution_request = await self.execution_queue.get()
                
                # Execute the trade
                result = await self._execute_trade(execution_request)
                
                # Update stats
                self._update_execution_stats(result)
                
            except Exception as e:
                logger.error(f"Error in execution worker: {str(e)}")
                await asyncio.sleep(1)
    
    async def _execute_trade(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual trade"""
        try:
            symbol = request['symbol']
            action = request['action']  # BUY/SELL
            volume = request['volume']
            sl = request.get('sl', 0.0)
            tp = request.get('tp', 0.0)
            comment = request.get('comment', 'ARIA-DAN')
            magic = self._get_next_magic()
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {'success': False, 'error': f'No tick data for {symbol}'}
            
            # Prepare order
            if action == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = float(tick.ask)
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = float(tick.bid)
            
            # Create trade request
            trade_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            start_time = datetime.now()
            result = mt5.order_send(trade_request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                return {'success': False, 'error': f'Order failed: {error}'}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return {'success': False, 'error': result.comment}
            
            # Calculate slippage
            slippage = abs(result.price - price)
            
            logger.info(f"✅ Order executed: {symbol} {action} {volume} @ {result.price:.5f}")
            
            return {
                'success': True,
                'ticket': result.order,
                'symbol': symbol,
                'action': action,
                'volume': volume,
                'price': result.price,
                'slippage': slippage,
                'execution_time': execution_time,
                'magic': magic
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # PUBLIC API METHODS
    
    async def place_order(self, symbol: str, action: str, volume: float, 
                         sl: float = 0.0, tp: float = 0.0, comment: str = "ARIA-DAN") -> Dict[str, Any]:
        """Place market order"""
        try:
            # Validate parameters
            if not self._validate_order_params(symbol, action, volume):
                return {'success': False, 'error': 'Invalid order parameters'}
            
            # Risk check
            if not self._risk_check(symbol, volume):
                return {'success': False, 'error': 'Risk limits exceeded'}
            
            # Queue for execution
            execution_request = {
                'symbol': symbol,
                'action': action.upper(),
                'volume': volume,
                'sl': sl,
                'tp': tp,
                'comment': comment,
                'timestamp': datetime.now()
            }
            
            await self.execution_queue.put(execution_request)
            
            logger.info(f"📋 Order queued: {symbol} {action} {volume}")
            return {'success': True, 'message': 'Order queued for execution'}
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close specific position"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Position not found'}
            
            pos = position[0]
            
            # Create close request
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "deviation": 20,
                "magic": pos.magic,
                "comment": f"Close by ARIA-DAN",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Execute close
            result = mt5.order_send(close_request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = result.comment if result else mt5.last_error()
                logger.error(f"Failed to close position {ticket}: {error}")
                return {'success': False, 'error': f'Close failed: {error}'}
            
            logger.info(f"✅ Position closed: {ticket} @ {result.price:.5f}")
            
            return {
                'success': True,
                'ticket': ticket,
                'close_price': result.price,
                'profit': pos.profit
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return {'success': True, 'message': 'No positions to close'}
            
            results = []
            for pos in positions:
                result = await self.close_position(pos.ticket)
                results.append(result)
            
            successful = len([r for r in results if r['success']])
            
            logger.info(f"📊 Closed {successful}/{len(results)} positions")
            
            return {
                'success': True,
                'total_positions': len(results),
                'successful_closes': successful,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error closing all positions: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_live_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current live price for symbol"""
        return self.live_prices.get(symbol)
    
    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            # Map timeframe
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            if rates is None:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not self.account_info:
                return {}
            
            return {
                'login': self.account_info.login,
                'balance': float(self.account_info.balance),
                'equity': float(self.account_info.equity),
                'margin': float(self.account_info.margin),
                'free_margin': float(self.account_info.margin_free),
                'margin_level': float(self.account_info.margin_level),
                'currency': self.account_info.currency,
                'trade_allowed': self.account_info.trade_allowed,
                'profit': float(self.account_info.profit)
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        try:
            positions = []
            for ticket, pos in self.open_positions.items():
                mt5_pos = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol, 
                    'position_type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': float(pos.profit),
                    'swap': float(getattr(pos, 'swap', 0.0)),
                    'commission': 0.0,  # Commission not available in position object
                    'timestamp': datetime.fromtimestamp(pos.time)
                }       
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    # HELPER METHODS
    
    def _get_monitored_symbols(self) -> List[str]:
        """Get list of symbols to monitor"""
        # Major forex pairs and commodities
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'XAUUSD', 'XAGUSD'
        ]
    
    def _get_next_magic(self) -> int:
        """Generate next magic number"""
        self.magic_counter += 1
        return self.aria_magic_base + self.magic_counter
    
    def _validate_order_params(self, symbol: str, action: str, volume: float) -> bool:
        """Validate order parameters"""
        try:
            # Check symbol
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Invalid symbol: {symbol}")
                return False
            
            # Check action
            if action.upper() not in ['BUY', 'SELL']:
                logger.error(f"Invalid action: {action}")
                return False
            
            # Check volume
            if volume <= 0:
                logger.error(f"Invalid volume: {volume}")
                return False
            
            if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
                logger.error(f"Volume {volume} outside limits: {symbol_info.volume_min}-{symbol_info.volume_max}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order params: {str(e)}")
            return False
    
    def _risk_check(self, symbol: str, volume: float) -> bool:
        """Check risk limits"""
        try:
            # Check max positions
            if len(self.open_positions) >= self.max_open_positions:
                logger.warning(f"Maximum positions limit reached: {self.max_open_positions}")
                return False
            
            # Check account balance
            if not self.account_info:
                return False
            
            # Simple risk check - ensure we have enough margin
            required_margin = volume * 100000 * 0.01  # Simplified margin calc
            if required_margin > self.account_info.margin_free:
                logger.warning(f"Insufficient margin: required {required_margin}, available {self.account_info.margin_free}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {str(e)}")
            return False
    
    def _update_execution_stats(self, result: Dict[str, Any]):
        """Update execution statistics"""
        try:
            self.execution_stats['total_orders'] += 1
            
            if result['success']:
                self.execution_stats['successful_orders'] += 1
                self.execution_stats['slippage_total'] += result.get('slippage', 0.0)
                
                # Update average execution time
                exec_time = result.get('execution_time', 0.0)
                total_successful = self.execution_stats['successful_orders']
                current_avg = self.execution_stats['avg_execution_time']
                
                # Calculate new average
                self.execution_stats['avg_execution_time'] = (
                    (current_avg * (total_successful - 1)) + exec_time
                ) / total_successful
                
            else:
                self.execution_stats['failed_orders'] += 1
                
        except Exception as e:
            logger.error(f"Error updating execution stats: {str(e)}")
    
    async def _notify_tick_subscribers(self, symbol: str, price_data: Dict[str, Any]):
        """Notify tick subscribers"""
        try:
            for subscriber in self.tick_subscribers:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(symbol, price_data)
                else:
                    subscriber(symbol, price_data)
                    
        except Exception as e:
            logger.error(f"Error notifying subscribers: {str(e)}")
    
    def subscribe_to_ticks(self, callback):
        """Subscribe to tick updates"""
        self.tick_subscribers.append(callback)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.execution_stats.copy()
        
        # Calculate success rate
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
        else:
            stats['success_rate'] = 0.0
        
        # Calculate average slippage
        if stats['successful_orders'] > 0:
            stats['avg_slippage'] = stats['slippage_total'] / stats['successful_orders']
        else:
            stats['avg_slippage'] = 0.0
        
        return stats
    
    async def shutdown(self):
        """Shutdown MT5 bridge"""
        try:
            logger.info("🔄 Shutting down MT5 Bridge")
            
            self.is_connected = False
            
            # Cleanup MT5
            mt5.shutdown()
            
            logger.info("✅ MT5 Bridge shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
