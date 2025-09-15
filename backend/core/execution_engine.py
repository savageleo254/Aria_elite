import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from utils.discord_notifier import discord_notifier
from .mt5_bridge import MT5Bridge

logger = setup_logger(__name__)

class ExecutionEngine:
    """
    Handles trade execution, risk management, and position monitoring
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.active_trades = {}
        self.trade_history = []
        self.last_execution_time = None
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_slippage': 0.0,
            'avg_execution_time': 0.0,
            "avg_slippage": 0.0
        }
        self.mt5_bridge = None
        self.is_emergency_mode = False
        
        # Risk management parameters (will be loaded from config)
        self.max_position_size = 1.0
        self.max_daily_loss = 1000.0
        self.max_risk_per_trade = 0.02
        
    async def initialize(self):
        """Initialize the execution engine"""
        try:
            logger.info("Initializing Execution Engine")
            
            # Load configuration
            await self._load_config()
            
            # Connect to MT5 via bridge
            await self._initialize_mt5_bridge()
            
            # Start monitoring loops
            asyncio.create_task(self._position_monitoring_loop())
            asyncio.create_task(self._execution_stats_loop())
            
            self.is_initialized = True
            logger.info("Execution Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Execution Engine: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load execution configuration"""
        try:
            self.execution_config = self.config.load_execution_config()
            self.project_config = self.config.load_project_config()
            
            logger.info("Execution configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load execution configuration: {str(e)}")
            raise
    
    async def _initialize_mt5_bridge(self):
        """Initialize MT5 bridge connection"""
        try:
            self.mt5_bridge = MT5Bridge()
            await self.mt5_bridge.initialize()
            
            # Validate MT5 connection
            if not self.mt5_bridge.is_connected:
                raise Exception("MT5 bridge is not connected")
            
            # Get account information
            account_info = self.mt5_bridge.get_account_info()
            if not account_info:
                error_msg = "Failed to retrieve MT5 account information"
                logger.error(error_msg)
                await discord_notifier.send_critical_error(
                    error=error_msg,
                    component="ExecutionEngine._initialize_mt5_bridge"
                )
                raise ConnectionError(error_msg)
            
            logger.info(f"MT5 Bridge connected successfully")
            logger.info(f"Account: {account_info.get('login')}, Server: {account_info.get('server')}")
            logger.info(f"Balance: {account_info.get('balance')}, Equity: {account_info.get('equity')}")
            
        except ImportError as e:
            error_msg = f"MetaTrader5 package not installed: {str(e)}"
            logger.error(error_msg)
            await discord_notifier.send_critical_error(
                error=error_msg,
                component="ExecutionEngine._initialize_mt5_bridge"
            )
            raise ImportError(f"{error_msg}. Install with: pip install MetaTrader5")
            
        except Exception as e:
            error_msg = f"Failed to initialize MT5 bridge: {str(e)}"
            logger.error(error_msg)
            await discord_notifier.send_critical_error(
                error=error_msg,
                component="ExecutionEngine._initialize_mt5_bridge"
            )
            raise
    
    async def execute_trade(
        self,
        signal_id: str,
        volume: float,
        order_type: str = "market",
        slippage_tolerance: float = 0.0005
    ) -> Dict[str, Any]:
        """
        Execute a trade based on signal
        
        Args:
            signal_id: ID of the approved signal
            volume: Trade volume
            order_type: Type of order (market, limit, stop)
            slippage_tolerance: Maximum acceptable slippage
            
        Returns:
            Trade execution result
        """
        try:
            logger.info(f"Executing trade for signal {signal_id}")
            
            # Get signal details (this would normally come from database)
            signal = await self._get_signal_details(signal_id)
            if not signal:
                raise ValueError(f"Signal {signal_id} not found")
            
            # Validate trade parameters
            await self._validate_trade_parameters(signal, volume)
            
            # Execute trade via MT5 bridge
            execution_result = await self.mt5_bridge.place_order(
                symbol=signal["symbol"],
                order_type=order_type,
                volume=volume,
                price=signal["entry_price"],
                sl=signal.get("stop_loss"),
                tp=signal.get("take_profit"),
                slippage=int(slippage_tolerance * 100000),
                comment=f"ARIA-ELITE-{signal_id}"
            )
            
            # Record trade
            trade_record = await self._create_trade_record(signal, execution_result, volume)
            
            # Add to active trades
            self.active_trades[trade_record["trade_id"]] = trade_record
            
            # Update statistics
            await self._update_execution_stats(execution_result)
            
            logger.info(f"Trade executed successfully: {trade_record['trade_id']}")
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise
    
    async def _get_signal_details(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get signal details from database or cache"""
        try:
            # Check active signals from signal manager first
            if hasattr(self, 'signal_manager') and self.signal_manager:
                active_signals = await self.signal_manager.get_active_signals()
                for signal in active_signals:
                    if signal.get('signal_id') == signal_id:
                        return signal
            
            # Fallback: check signal history
            if hasattr(self, 'signal_manager') and self.signal_manager:
                signal_history = await self.signal_manager.get_signal_history(limit=1000)
                for signal in signal_history:
                    if signal.get('signal_id') == signal_id:
                        return signal
            
            logger.warning(f"Signal {signal_id} not found in signal manager")
            return None
            
        except Exception as e:
            logger.error(f"Error getting signal details: {str(e)}")
            return None
    
    async def _validate_trade_parameters(self, signal: Dict[str, Any], volume: float):
        """Validate trade parameters"""
        try:
            # Check volume
            if volume <= 0:
                raise ValueError("Volume must be positive")
            
            # Check risk parameters
            risk_amount = abs(signal["entry_price"] - signal["stop_loss"]) * volume
            if risk_amount > 100:  # Max risk amount
                raise ValueError("Risk amount exceeds maximum allowed")
            
            # Check if market is open (simplified)
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 18:
                raise ValueError("Market is closed")
            
            logger.info("Trade parameters validated successfully")
            
        except Exception as e:
            logger.error(f"Error validating trade parameters: {str(e)}")
            raise
    
    async def _execute_order_legacy(self, signal: Dict[str, Any], volume: float, order_type: str, slippage_tolerance: float) -> Dict[str, Any]:
        """Execute order on MT5 via bridge"""
        if not self.mt5_bridge:
            error_msg = "MT5 Bridge not initialized - cannot execute orders"
            logger.error(error_msg)
            await discord_notifier.send_critical_error(
                error=error_msg,
                component="ExecutionEngine._execute_order_legacy"
            )
            raise ConnectionError(error_msg)
        
        try:
            start_time = datetime.now()
            
            # Prepare order request
            order_request = {
                "symbol": signal["symbol"],
                "action": "BUY" if signal["direction"].upper() == "BUY" else "SELL",
                "volume": volume,
                "price": signal["entry_price"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "order_type": order_type,
                "slippage": int(slippage_tolerance * 100000),  # Convert to points
                "comment": f"ARIA_SIGNAL_{signal['signal_id']}"
            }
            
            # Execute order via MT5 Bridge
            execution_result = await self.mt5_bridge.send_order(order_request)
            
            if not execution_result or not execution_result.get("success"):
                error_msg = execution_result.get("error", "Unknown execution error")
                logger.error(f"Order execution failed: {error_msg}")
                await discord_notifier.send_trading_alert(
                    alert_type="execution",
                    symbol=signal["symbol"],
                    message=f"Order execution failed: {error_msg}"
                )
                
                return {
                    "success": False,
                    "execution_price": 0,
                    "requested_price": signal["entry_price"],
                    "slippage": 0,
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "order_id": None,
                    "timestamp": datetime.now(),
                    "error_message": error_msg
                }
            
            # Calculate slippage
            execution_price = execution_result.get("price", signal["entry_price"])
            slippage = abs(execution_price - signal["entry_price"])
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "execution_price": execution_price,
                "requested_price": signal["entry_price"],
                "slippage": slippage,
                "execution_time": execution_time,
                "order_id": execution_result.get("order_id"),
                "timestamp": datetime.now(),
                "error_message": None
            }
            
            logger.info(f"Order executed successfully: ID={result['order_id']}, Price={execution_price}, Slippage={slippage}")
            
            # Send success notification
            await discord_notifier.send_trading_alert(
                alert_type="execution",
                symbol=signal["symbol"],
                message=f"Order executed: {signal['direction']} {volume} lots at {execution_price}",
                signal_data={"direction": signal["direction"], "confidence": signal.get("confidence", 0.5)}
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Critical error executing order: {str(e)}"
            logger.error(error_msg)
            await discord_notifier.send_critical_error(
                error=error_msg,
                component="ExecutionEngine._execute_order_legacy"
            )
            
            return {
                "success": False,
                "execution_price": 0,
                "requested_price": signal["entry_price"],
                "slippage": 0,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "order_id": None,
                "timestamp": datetime.now(),
                "error_message": error_msg
            }
    
    async def _create_trade_record(
        self,
        signal: Dict[str, Any],
        execution_result: Dict[str, Any],
        volume: float
    ) -> Dict[str, Any]:
        """Create trade record"""
        try:
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal['symbol']}"
            
            trade_record = {
                "trade_id": trade_id,
                "signal_id": signal["signal_id"],
                "symbol": signal["symbol"],
                "direction": signal["direction"],
                "volume": volume,
                "entry_price": execution_result["execution_price"],
                "requested_price": execution_result["requested_price"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "order_type": "market",
                "order_id": execution_result["order_id"],
                "status": "open" if execution_result["success"] else "failed",
                "open_time": datetime.now() if execution_result["success"] else None,
                "close_time": None,
                "close_price": None,
                "pnl": 0.0,
                "slippage": execution_result["slippage"],
                "execution_time": execution_result["execution_time"],
                "error_message": execution_result["error_message"]
            }
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error creating trade record: {str(e)}")
            raise
    
    async def _update_execution_stats(self, execution_result: Dict[str, Any]):
        """Update execution statistics"""
        try:
            self.execution_stats["total_trades"] += 1
            
            if execution_result["success"]:
                self.execution_stats["successful_trades"] += 1
            else:
                self.execution_stats["failed_trades"] += 1
            
            # Update average execution time
            total_time = (
                self.execution_stats["avg_execution_time"] * 
                (self.execution_stats["total_trades"] - 1) + 
                execution_result["execution_time"]
            )
            self.execution_stats["avg_execution_time"] = total_time / self.execution_stats["total_trades"]
            
            # Update slippage statistics
            self.execution_stats["total_slippage"] += execution_result["slippage"]
            self.execution_stats["avg_slippage"] = (
                self.execution_stats["total_slippage"] / self.execution_stats["total_trades"]
            )
            
        except Exception as e:
            logger.error(f"Error updating execution stats: {str(e)}")
    
    async def _position_monitoring_loop(self):
        """Monitor open positions"""
        while True:
            try:
                if not self.is_emergency_mode:
                    await self._monitor_positions()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _monitor_positions(self):
        """Monitor and update open positions"""
        try:
            for trade_id, trade in list(self.active_trades.items()):
                if trade["status"] == "open":
                    await self._update_position_status(trade_id, trade)
                    
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
    
    async def _update_position_status(self, trade_id: str, trade: Dict[str, Any]):
        """Update status of a single position using live MT5 data"""
        if not self.mt5_bridge:
            error_msg = "MT5 Bridge not available - cannot monitor positions"
            logger.error(error_msg)
            return
        
        try:
            # Get current price
            current_price = await self.mt5_bridge.get_live_price(trade['symbol'])
            
            if not current_price:
                raise Exception(f"Cannot get current price for {trade['symbol']}")
            
            # Extract bid/ask from price data
            if isinstance(current_price, dict):
                price_value = current_price.get('bid', current_price.get('ask', current_price.get('price', 0)))
            else:
                price_value = float(current_price)
            
            # Check position status in MT5
            mt5_position = await self.mt5_bridge.get_position_by_ticket(trade.get("order_id"))
            if not mt5_position:
                # Position was closed externally or doesn't exist
                logger.warning(f"Position {trade_id} not found in MT5 - marking as closed")
                await self._close_position(trade_id, price_value, "external_close")
                return
            
            # Update trade with current MT5 position data
            trade["current_price"] = price_value
            trade["unrealized_pnl"] = mt5_position.get("profit", 0)
            
            # Check if position was closed by SL/TP in MT5
            if mt5_position.get("reason") in ["DEAL_REASON_SL", "DEAL_REASON_TP"]:
                close_reason = "stop_loss" if mt5_position.get("reason") == "DEAL_REASON_SL" else "take_profit"
                await self._close_position(trade_id, price_value, close_reason)
                
        except Exception as e:
            error_msg = f"Error updating position status for {trade_id}: {str(e)}"
            logger.error(error_msg)
            await discord_notifier.send_audit_alert(
                severity="high",
                component="ExecutionEngine._update_position_status",
                issue=error_msg
            )
    
    async def _close_position(self, trade_id: str, close_price: float, reason: str):
        """Close a position"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade or trade["status"] != "open":
                return
            
            # Calculate PnL
            if trade["direction"] == "BUY":
                pnl = (close_price - trade["entry_price"]) * trade["volume"]
            else:  # SELL
                pnl = (trade["entry_price"] - close_price) * trade["volume"]
            
            # Update trade record
            trade["status"] = "closed"
            trade["close_time"] = datetime.now()
            trade["close_price"] = close_price
            trade["pnl"] = pnl
            trade["close_reason"] = reason
            
            # Move to history
            self.trade_history.append(trade.copy())
            del self.active_trades[trade_id]
            
            logger.info(f"Position closed: {trade_id}, PnL: {pnl}, Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    async def _execution_stats_loop(self):
        """Update execution statistics periodically"""
        while True:
            try:
                await self._cleanup_old_trades()
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error in execution stats loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_trades(self):
        """Clean up old trade records"""
        try:
            # Keep only last 1000 trades in history
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error cleaning up old trades: {str(e)}")
    
    async def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades"""
        try:
            return list(self.active_trades.values())
            
        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            return []
    
    async def get_trade_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get trade history with pagination"""
        try:
            history = self.trade_history[-(limit + offset):]
            if offset > 0:
                history = history[offset:]
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            stats = self.execution_stats.copy()
            
            # Calculate additional metrics
            if stats["total_trades"] > 0:
                stats["success_rate"] = stats["successful_trades"] / stats["total_trades"]
                stats["failure_rate"] = stats["failed_trades"] / stats["total_trades"]
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
            
            # Calculate total PnL from history
            total_pnl = sum(trade.get("pnl", 0) for trade in self.trade_history)
            stats["total_pnl"] = total_pnl
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting execution stats: {str(e)}")
            return {}
    
    async def emergency_close_all(self):
        """Emergency close all positions"""
        try:
            logger.warning("Emergency close all positions activated")
            self.is_emergency_mode = True
            
            # Close all active positions via MT5
            if not self.mt5_bridge:
                error_msg = "MT5 Bridge not available - cannot execute emergency close"
                logger.error(error_msg)
                await discord_notifier.send_critical_error(
                    error=error_msg,
                    component="ExecutionEngine.emergency_close_all"
                )
                raise ConnectionError(error_msg)
            
            for trade_id, trade in list(self.active_trades.items()):
                if trade["status"] == "open":
                    try:
                        # Get all positions and find by ticket
                        positions = await self.mt5_bridge.get_positions()
                        position = None
                        
                        for pos in positions:
                            if pos.get('ticket') == trade.get("order_id"):
                                position = pos
                                break
                        
                        if not position:
                            logger.warning(f"Position with ticket {trade.get('order_id')} not found")
                            continue
                        
                        # Close position
                        close_result = await self.mt5_bridge.close_position(
                            ticket=trade.get("order_id"),
                            volume=position.get('volume'),
                            symbol=position.get('symbol')
                        )
                        
                        if close_result and close_result.get("success"):
                            close_price = close_result.get("price", trade["entry_price"])
                            await self._close_position(trade_id, close_price, "emergency")
                        else:
                            error_msg = f"Failed to close position {trade_id}: {close_result.get('error', 'Unknown error')}"
                            logger.error(error_msg)
                    except Exception as e:
                        logger.error(f"Error closing position {trade_id} during emergency: {str(e)}")
            
            logger.info("All positions closed due to emergency")
            
        except Exception as e:
            logger.error(f"Error in emergency close all: {str(e)}")
            raise
    
    async def close_position(self, trade_id: str, reason: str = "manual"):
        """Close a specific position"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade or trade["status"] != "open":
                raise ValueError(f"Trade {trade_id} not found or not open")
            
            if not self.mt5_bridge:
                error_msg = "MT5 Bridge not available - cannot close position"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
            
            # Close position via MT5 Bridge
            close_result = await self.mt5_bridge.close_position_by_ticket(trade.get("order_id"))
            
            if close_result and close_result.get("success"):
                close_price = close_result.get("price", trade["entry_price"])
                await self._close_position(trade_id, close_price, reason)
            else:
                error_msg = f"Failed to close position {trade_id}: {close_result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"Position {trade_id} closed manually")
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current execution engine status"""
        try:
            account_info = {}
            if self.mt5_bridge:
                account_info = self.mt5_bridge.get_account_info()
            
            return {
                "engine_status": "running" if self.is_initialized else "stopped",
                "mt5_connected": bool(self.mt5_bridge and self.mt5_bridge.is_connected),
                "active_trades": len([t for t in self.active_trades.values() if t["status"] == "open"]),
                "pending_orders": len([t for t in self.active_trades.values() if t["status"] == "pending"]),
                "account_balance": account_info.get("balance", 0.0),
                "account_equity": account_info.get("equity", 0.0),
                "account_login": account_info.get("login", "N/A"),
                "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
                "total_trades": len(self.active_trades),
                "risk_limits": {
                    "max_position_size": self.max_position_size,
                    "max_daily_loss": self.max_daily_loss,
                    "max_risk_per_trade": self.max_risk_per_trade
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting execution engine status: {str(e)}")
            return {"engine_status": "error", "error": str(e)}
    
    async def modify_position(self, trade_id: str, stop_loss: float = None, take_profit: float = None):
        """Modify stop loss or take profit for a position"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade or trade["status"] != "open":
                raise ValueError(f"Trade {trade_id} not found or not open")
            
            if stop_loss is not None:
                trade["stop_loss"] = stop_loss
            if take_profit is not None:
                trade["take_profit"] = take_profit
            
            logger.info(f"Position {trade_id} modified")
            
        except Exception as e:
            logger.error(f"Error modifying position: {str(e)}")
            raise
