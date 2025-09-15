#!/usr/bin/env python3
"""
ARIA Elite Live Trading System Launcher
Production-grade startup script for institutional trading deployment
"""

import asyncio
import logging
import os
import sys
import signal
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.signal_manager import SignalManager
from backend.core.execution_engine import ExecutionEngine
from backend.core.mt5_bridge import MT5Bridge
from backend.core.browser_ai_agent import BrowserAIManager
from backend.models.ai_models import AIModelManager
from backend.core.gemini_workflow_agent import MultiAIWorkflowAgent
from backend.utils.config_loader import ConfigLoader
from backend.utils.logger import setup_logger
from backend.utils.discord_notifier import discord_notifier

logger = setup_logger(__name__)

class ARIALiveTradingSystem:
    """
    ARIA Elite Live Trading System - Institutional Grade Deployment
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.signal_manager: Optional[SignalManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.mt5_bridge: Optional[MT5Bridge] = None
        self.browser_ai_manager: Optional[BrowserAIManager] = None
        self.ai_model_manager: Optional[AIModelManager] = None
        self.workflow_agent: Optional[MultiAIWorkflowAgent] = None
        
        # System state
        self.start_time = datetime.now()
        self.components_initialized = []
        self.failed_components = []
        
    async def initialize_system(self):
        """Initialize all trading system components"""
        logger.info("🚀 ARIA ELITE LIVE TRADING SYSTEM - INITIALIZING")
        logger.info("=" * 60)
        
        try:
            # 1. Initialize AI Model Manager first
            logger.info("📊 Initializing AI Model Manager...")
            self.ai_model_manager = AIModelManager()
            await self.ai_model_manager.initialize()
            self.components_initialized.append("AI Models")
            logger.info("✅ AI Models loaded and ready")
            
            # 2. Initialize MT5 Bridge
            logger.info("📈 Initializing MT5 Bridge...")
            self.mt5_bridge = MT5Bridge()
            await self.mt5_bridge.initialize()
            if not self.mt5_bridge.is_connected:
                raise Exception("MT5 Bridge failed to connect")
            self.components_initialized.append("MT5 Bridge")
            logger.info("✅ MT5 Bridge connected and operational")
            
            # 3. Initialize Browser AI Manager
            logger.info("🤖 Initializing Multi-AI Browser Manager...")
            self.browser_ai_manager = BrowserAIManager()
            try:
                await self.browser_ai_manager.initialize()
                self.components_initialized.append("Multi-AI Browser")
                logger.info("✅ Multi-AI Browser system operational")
            except Exception as e:
                logger.warning(f"⚠️ Multi-AI Browser initialization failed: {e}")
                logger.info("System will continue with limited AI capabilities")
                self.failed_components.append("Multi-AI Browser")
            
            # 4. Initialize Execution Engine
            logger.info("⚡ Initializing Execution Engine...")
            self.execution_engine = ExecutionEngine()
            await self.execution_engine.initialize()
            self.components_initialized.append("Execution Engine")
            logger.info("✅ Execution Engine ready for live trading")
            
            # 5. Initialize Signal Manager
            logger.info("📡 Initializing Signal Manager...")
            self.signal_manager = SignalManager()
            await self.signal_manager.initialize()
            self.components_initialized.append("Signal Manager")
            logger.info("✅ Signal Manager operational")
            
            # 6. Initialize Workflow Agent
            logger.info("🧠 Initializing Multi-AI Workflow Agent...")
            self.workflow_agent = MultiAIWorkflowAgent()
            await self.workflow_agent.initialize()
            self.components_initialized.append("Workflow Agent")
            logger.info("✅ Workflow Agent ready for autonomous operations")
            
            # System initialization complete
            self.is_running = True
            logger.info("=" * 60)
            logger.info("🎯 ARIA ELITE LIVE TRADING SYSTEM - FULLY OPERATIONAL")
            logger.info(f"✅ Components initialized: {', '.join(self.components_initialized)}")
            if self.failed_components:
                logger.warning(f"⚠️ Failed components: {', '.join(self.failed_components)}")
            logger.info("=" * 60)
            
            # Send Discord notification
            await discord_notifier.send_system_startup(
                components_initialized=self.components_initialized,
                failed_components=self.failed_components,
                start_time=self.start_time
            )
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: System initialization failed: {str(e)}")
            raise
    
    async def run_trading_loop(self):
        """Main trading loop - monitors and coordinates all trading activities"""
        logger.info("🔄 Starting main trading loop...")
        
        try:
            while not self.shutdown_event.is_set():
                # System health check
                await self._system_health_check()
                
                # Process any pending signals
                if self.signal_manager and not self.signal_manager.is_paused:
                    await self._process_signals()
                
                # Monitor positions and risk
                if self.execution_engine:
                    await self._monitor_positions()
                
                # AI workflow tasks
                if self.workflow_agent:
                    await self._process_workflow_tasks()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5-second loop
                
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {str(e)}")
            await self._emergency_shutdown()
    
    async def _system_health_check(self):
        """Perform system health checks"""
        try:
            # Check MT5 connection
            if self.mt5_bridge and not self.mt5_bridge.is_connected:
                logger.warning("⚠️ MT5 connection lost - attempting reconnection...")
                await self.mt5_bridge.initialize()
            
            # Check memory usage
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"⚠️ High memory usage: {memory_percent}%")
                
        except Exception as e:
            logger.error(f"❌ Health check error: {str(e)}")
    
    async def _process_signals(self):
        """Process pending trading signals"""
        try:
            # This would normally check for new signals and process them
            # Implementation depends on your specific signal generation logic
            pass
        except Exception as e:
            logger.error(f"❌ Signal processing error: {str(e)}")
    
    async def _monitor_positions(self):
        """Monitor open positions and risk"""
        try:
            if self.execution_engine:
                # Monitor open positions for stop-loss/take-profit
                await self.execution_engine._position_monitoring_loop()
        except Exception as e:
            logger.error(f"❌ Position monitoring error: {str(e)}")
    
    async def _process_workflow_tasks(self):
        """Process AI workflow agent tasks"""
        try:
            if self.workflow_agent:
                # Process any pending workflow tasks
                pass
        except Exception as e:
            logger.error(f"❌ Workflow processing error: {str(e)}")
    
    async def _emergency_shutdown(self):
        """Emergency system shutdown"""
        logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Close all positions
            if self.execution_engine:
                await self.execution_engine.emergency_close_all()
            
            # Pause signal generation
            if self.signal_manager:
                await self.signal_manager.pause()
            
            # Disconnect MT5
            if self.mt5_bridge:
                await self.mt5_bridge.disconnect()
            
            # Send critical alert
            await discord_notifier.send_critical_error(
                error="Emergency shutdown initiated",
                component="LiveTradingSystem"
            )
            
        except Exception as e:
            logger.error(f"❌ Error during emergency shutdown: {str(e)}")
        
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Graceful system shutdown"""
        logger.info("🛑 Initiating graceful system shutdown...")
        
        try:
            # Stop trading loop
            self.shutdown_event.set()
            
            # Close all positions gracefully
            if self.execution_engine:
                await self.execution_engine.close_all_positions()
            
            # Stop all components
            components = [
                self.workflow_agent,
                self.signal_manager,
                self.execution_engine,
                self.browser_ai_manager,
                self.mt5_bridge
            ]
            
            for component in components:
                if component and hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                    except Exception as e:
                        logger.warning(f"⚠️ Component shutdown error: {str(e)}")
            
            logger.info("✅ ARIA Elite system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")

# Global system instance
trading_system: Optional[ARIALiveTradingSystem] = None

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    global trading_system
    logger.info(f"📡 Received signal {signum} - initiating shutdown...")
    
    if trading_system:
        asyncio.create_task(trading_system.shutdown())

async def main():
    """Main entry point for ARIA Elite Live Trading System"""
    global trading_system
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        trading_system = ARIALiveTradingSystem()
        await trading_system.initialize_system()
        
        # Start trading loop
        await trading_system.run_trading_loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"❌ CRITICAL SYSTEM ERROR: {str(e)}")
        raise
    finally:
        if trading_system:
            await trading_system.shutdown()

if __name__ == "__main__":
    # Set UTF-8 encoding for stdout
    sys.stdout.reconfigure(encoding='utf-8')
    print("🚀 ARIA ELITE LIVE TRADING SYSTEM")
    print("=" * 50)
    print("Institutional-Grade Autonomous Trading Platform")
    print("Starting system initialization...")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        sys.exit(1)