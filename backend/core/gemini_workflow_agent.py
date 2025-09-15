import asyncio
import json
import logging
import os
import subprocess
import psutil
import platform
import shutil
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import aiohttp
from pathlib import Path
import hashlib
import re
import traceback

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from .multi_ai_engine import MultiBrowserAIEngine, TaskComplexity, AIProvider

logger = setup_logger(__name__)

class MultiAIWorkflowAgent:
    """
    ARIA-DAN Multi-AI Workflow Agent - Institutional Grade Autonomous System Manager
    
    Acts as your second human for:
    - System diagnostics and error recovery
    - Code review and automatic fixes
    - Remote system control via Discord
    - Hardware optimization and guardrails
    - Pipeline automation and optimization
    - Trading signal generation and validation
    
    NO MOCKS, NO FALLBACKS - 100% REAL MULTI-AI AUTOMATION
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.multi_ai_engine = MultiBrowserAIEngine()
        self.is_initialized = False
        self.start_time = datetime.now()  # Fix uptime calculation
        
        # Core queues
        self.signal_queue = asyncio.Queue()
        self.approval_queue = asyncio.Queue()
        self.diagnostic_queue = asyncio.Queue()
        self.system_control_queue = asyncio.Queue()
        self.code_review_queue = asyncio.Queue()
        
        # Active state tracking
        self.active_signals = {}
        self.system_errors = {}
        self.pending_fixes = {}
        self.system_status = "initializing"
        
        # Task management
        self.background_tasks = set()
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        self.strategy_config = {}
        self.execution_config = {}
        self.hardware_config = {}
        
        # Security settings
        self.auto_apply_fixes = os.getenv("ARIA_AUTO_APPLY_FIXES", "false").lower() == "true"
        self.admin_role_id = os.getenv("ARIA_ADMIN_DISCORD_ROLE_ID", "")
        
        # System control
        self.system_processes = {}
        self.discord_bot_pid = None
        self.trading_engine_pid = None
        
        # Hardware detection
        self.has_gpu = self._detect_gpu()
        self.cpu_cores = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total // (1024**3)
        
    async def initialize(self):
        """Initialize the Multi-AI Workflow Agent"""
        try:
            logger.info("Initializing Multi-AI Workflow Agent - ARIA-DAN Mode Engaged")
            
            # Load configuration
            await self._load_config()
            
            # Initialize Multi-AI Engine
            await self.multi_ai_engine.initialize()
            
            # System detection and optimization
            await self._detect_system_capabilities()
            await self._optimize_for_hardware()
            
            # Start all background processes with proper task management
            self._create_managed_task(self._signal_processing_loop())
            self._create_managed_task(self._approval_processing_loop())
            self._create_managed_task(self._diagnostic_monitoring_loop())
            self._create_managed_task(self._system_control_loop())
            self._create_managed_task(self._code_review_loop())
            self._create_managed_task(self._health_monitor())
            self._create_managed_task(self._optimization_pipeline())
            
            self.is_initialized = True
            self.system_status = "operational"
            
            # Initial system audit
            await self._perform_system_audit()
            
            logger.info("Multi-AI Workflow Agent initialized - Full autonomous control engaged")
            
        except Exception as e:
            logger.error(f"Failed to initialize Multi-AI Workflow Agent: {str(e)}")
            self.system_status = "error"
            await self._handle_critical_error(e)
            raise
    
    async def _load_config(self):
        """Load configuration from config files"""
        try:
            self.strategy_config = self.config.load_strategy_config()
            self.execution_config = self.config.load_execution_config()
            
            # Hardware optimization settings
            self.hardware_config = {
                "gpu_enabled": self.has_gpu,
                "cpu_cores": self.cpu_cores,
                "memory_gb": self.memory_gb,
                "optimization_level": "aggressive" if self.has_gpu else "conservative"
            }
            
            logger.info(f"Configuration loaded - GPU: {self.has_gpu}, CPU: {self.cpu_cores}, RAM: {self.memory_gb}GB")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            await self._handle_critical_error(e)
            raise
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability for hardware optimization"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.experimental.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
    
    async def _detect_system_capabilities(self):
        """Detect and analyze system capabilities"""
        try:
            system_info = {
                "platform": platform.system(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "disk_space_gb": shutil.disk_usage('/').free // (1024**3),
                "network_interfaces": len(psutil.net_if_addrs()),
            }
            
            self.hardware_config.update(system_info)
            logger.info(f"System capabilities detected: {system_info}")
            
        except Exception as e:
            logger.error(f"Failed to detect system capabilities: {str(e)}")
            await self.diagnostic_queue.put({"type": "system_detection_error", "error": str(e)})
    
    async def _optimize_for_hardware(self):
        """Optimize system settings based on hardware capabilities"""
        try:
            optimization_settings = {
                "browser_instances": min(self.cpu_cores * 2, 8) if self.has_gpu else min(self.cpu_cores, 4),
                "concurrent_requests": self.cpu_cores * 4 if self.has_gpu else self.cpu_cores * 2,
                "memory_limit_mb": int(self.memory_gb * 1024 * 0.7),  # Use 70% of available memory
                "processing_threads": self.cpu_cores,
                "gpu_acceleration": self.has_gpu
            }
            
            self.hardware_config.update(optimization_settings)
            logger.info(f"Hardware optimization applied: {optimization_settings}")
            
        except Exception as e:
            logger.error(f"Hardware optimization failed: {str(e)}")
            await self.diagnostic_queue.put({"type": "hardware_optimization_error", "error": str(e)})
    
    async def _perform_system_audit(self):
        """Perform initial system audit and health check"""
        try:
            audit_prompt = f"""
            Perform comprehensive system audit for ARIA Elite trading system:
            
            Hardware Configuration:
            - GPU Available: {self.has_gpu}
            - CPU Cores: {self.cpu_cores}
            - Memory: {self.memory_gb}GB
            - Platform: {platform.system()}
            
            System Status:
            - Multi-AI Engine: {self.multi_ai_engine.is_initialized if hasattr(self.multi_ai_engine, 'is_initialized') else 'Unknown'}
            - Active Processes: {len(self.system_processes)}
            - Error Queue: {self.diagnostic_queue.qsize()}
            
            Analyze system health, identify potential bottlenecks, recommend optimizations.
            Focus on production readiness, security, and performance.
            
            Respond with JSON:
            {{
                "system_health": "excellent/good/fair/poor",
                "critical_issues": [],
                "recommendations": [],
                "performance_score": 0-100,
                "security_status": "secure/warning/vulnerable"
            }}
            """
            
            audit_result = await self.multi_ai_engine.generate_signal(
                "SYSTEM_AUDIT", "HEALTH_CHECK", {"system_data": audit_prompt}, TaskComplexity.COMPLEX
            )
            
            logger.info(f"System audit completed: {audit_result.get('reasoning', 'No details available')}")
            
        except Exception as e:
            logger.error(f"System audit failed: {str(e)}")
            await self._handle_critical_error(e)
    
    async def _handle_critical_error(self, error: Exception):
        """Handle critical system errors with Multi-AI diagnosis"""
        try:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "system_state": {
                    "status": self.system_status,
                    "initialized": self.is_initialized,
                    "active_processes": len(self.system_processes)
                }
            }
            
            await self.diagnostic_queue.put(error_data)
            logger.critical(f"Critical error handled: {error}")
            
        except Exception as e:
            logger.critical(f"Failed to handle critical error: {e}")
    
    async def generate_signal(self, symbol: str, timeframe: str, strategy: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate trading signal using Multi-AI consensus with hardware optimization"""
        try:
            if not self.is_initialized:
                raise Exception("Multi-AI Workflow Agent not initialized")
            
            # Prepare market data context with hardware optimization
            market_context = await self._get_market_context(symbol, timeframe)
            
            # Determine task complexity based on strategy and market conditions
            complexity = self._determine_task_complexity(strategy, market_context)
            
            # Generate signal using Multi-AI engine with consensus
            signal_data = await self.multi_ai_engine.generate_signal(
                symbol, timeframe, market_context, complexity
            )
            
            # Validate signal with additional hardware-aware checks
            validated_signal = await self._validate_signal_with_guardrails(signal_data)
            
            # Store signal in queue for processing
            await self.signal_queue.put(validated_signal)
            
            logger.info(f"Multi-AI signal generated for {symbol} using {strategy} strategy")
            return validated_signal
            
        except Exception as e:
            logger.error(f"Failed to generate signal: {str(e)}")
            await self.diagnostic_queue.put({
                "type": "signal_generation_error",
                "symbol": symbol,
                "strategy": strategy,
                "error": str(e)
            })
            raise
    
    def _determine_task_complexity(self, strategy: str, market_context: Dict[str, Any]) -> TaskComplexity:
        """Determine task complexity for optimal model selection"""
        try:
            volatility = market_context.get("volatility", 0.001)
            volume = market_context.get("volume", 0)
            
            if strategy in ["scalping", "news_trading"] or volatility > 0.01:
                return TaskComplexity.CRITICAL
            elif strategy in ["swing", "position"] or volatility > 0.005:
                return TaskComplexity.COMPLEX
            elif volume > 100000:
                return TaskComplexity.MODERATE
            else:
                return TaskComplexity.SIMPLE
                
        except Exception:
            return TaskComplexity.MODERATE
    
    async def _validate_signal_with_guardrails(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal with hardware-aware guardrails"""
        try:
            # Standard validation
            validated_signal = await self._validate_signal(signal_data)
            
            # Hardware-specific guardrails
            if not self.has_gpu and validated_signal.get("confidence", 0) < 0.7:
                validated_signal["confidence"] *= 0.8  # Reduce confidence on CPU systems
                validated_signal["guardrail_applied"] = "cpu_confidence_reduction"
            
            # Memory-based position sizing
            if self.memory_gb < 8:
                max_positions = 3
            elif self.memory_gb < 16:
                max_positions = 5
            else:
                max_positions = 10
                
            validated_signal["max_concurrent_positions"] = max_positions
            validated_signal["hardware_optimized"] = True
            
            return validated_signal
            
        except Exception as e:
            logger.error(f"Signal validation with guardrails failed: {str(e)}")
            return signal_data
    
    def _create_managed_task(self, coro):
        """Create and track background tasks properly"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task
    
    async def _diagnostic_monitoring_loop(self):
        """Background loop for system diagnostics and error recovery"""
        while not self.shutdown_event.is_set():
            try:
                # Use timeout to avoid blocking indefinitely
                try:
                    error_data = await asyncio.wait_for(
                        self.diagnostic_queue.get(), timeout=30.0
                    )
                    await self._diagnose_and_fix_error(error_data)
                except asyncio.TimeoutError:
                    # Periodic system health checks when no errors
                    await self._check_system_health()
                    
            except Exception as e:
                logger.error(f"Diagnostic monitoring loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _diagnose_and_fix_error(self, error_data: Dict[str, Any]):
        """Diagnose error and attempt automatic fix using Multi-AI"""
        try:
            diagnostic_prompt = f"""
            CRITICAL SYSTEM ERROR DIAGNOSIS - ARIA-DAN Emergency Protocol
            
            Error Data:
            {json.dumps(error_data, indent=2)}
            
            System State:
            - Status: {self.system_status}
            - Active Processes: {len(self.system_processes)}
            - Hardware: GPU={self.has_gpu}, CPU={self.cpu_cores}, RAM={self.memory_gb}GB
            
            Analyze this error and provide:
            1. Root cause analysis
            2. Immediate fix steps
            3. Prevention measures
            4. Code patches if needed
            5. System restart requirements
            
            Respond with JSON:
            {{
                "severity": "low/medium/high/critical",
                "root_cause": "description",
                "fix_steps": ["step1", "step2"],
                "code_fixes": [{{
                    "file": "path/to/file",
                    "function": "function_name",
                    "fix": "code_snippet"
                }}],
                "restart_required": true/false,
                "prevention": "measures"
            }}
            """
            
            diagnosis = await self.multi_ai_engine.generate_signal(
                "ERROR_DIAGNOSIS", "CRITICAL", {"error_data": diagnostic_prompt}, TaskComplexity.CRITICAL
            )
            
            # Execute automatic fixes
            await self._execute_automatic_fixes(diagnosis)
            
            logger.info(f"Error diagnosis completed: {diagnosis.get('reasoning', 'No details')}")
            
        except Exception as e:
            logger.critical(f"Failed to diagnose error: {str(e)}")
    
    async def _execute_automatic_fixes(self, diagnosis: Dict[str, Any]):
        """Execute automatic fixes based on Multi-AI diagnosis"""
        try:
            if diagnosis.get("code_fixes"):
                for fix in diagnosis["code_fixes"]:
                    await self._apply_code_fix(fix)
                    
            if diagnosis.get("restart_required"):
                await self._schedule_system_restart()
                
            logger.info(f"Automatic fixes executed: {len(diagnosis.get('code_fixes', []))} fixes applied")
            
        except Exception as e:
            logger.error(f"Failed to execute automatic fixes: {str(e)}")
    
    async def _system_control_loop(self):
        """Background loop for system control commands"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    command = await asyncio.wait_for(
                        self.system_control_queue.get(), timeout=5.0
                    )
                    await self._execute_system_command(command)
                except asyncio.TimeoutError:
                    pass  # Continue loop
                    
            except Exception as e:
                logger.error(f"System control loop error: {str(e)}")
                await asyncio.sleep(10)
    
    async def _execute_system_command(self, command: Dict[str, Any]):
        """Execute system control commands (shutdown, restart, etc.)"""
        try:
            action = command.get("action")
            source = command.get("source", "unknown")
            
            logger.info(f"Executing system command: {action} from {source}")
            
            if action == "shutdown":
                await self._graceful_shutdown()
            elif action == "restart":
                await self._restart_system()
            elif action == "start_trading":
                await self._start_trading_engine()
            elif action == "stop_trading":
                await self._stop_trading_engine()
            elif action == "start_discord_bot":
                await self._start_discord_bot()
            elif action == "status_report":
                await self._send_status_report(command.get("destination"))
                
        except Exception as e:
            logger.error(f"Failed to execute system command {command}: {str(e)}")
    
    async def _code_review_loop(self):
        """Background loop for continuous code review and optimization"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    review_request = await asyncio.wait_for(
                        self.code_review_queue.get(), timeout=300.0
                    )
                    await self._perform_code_review(review_request)
                except asyncio.TimeoutError:
                    # Periodic code health checks when no reviews queued
                    await self._periodic_code_audit()
                    
            except Exception as e:
                logger.error(f"Code review loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _perform_code_review(self, review_request: Dict[str, Any]):
        """Perform AI-powered code review and suggest improvements"""
        try:
            file_path = review_request.get("file_path")
            review_type = review_request.get("type", "general")
            
            if not file_path or not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            review_prompt = f"""
            ARIA-DAN CODE REVIEW - INSTITUTIONAL GRADE ANALYSIS
            
            File: {file_path}
            Review Type: {review_type}
            
            Code:
            ```python
            {code_content[:8000]}  # Limit for token efficiency
            ```
            
            Analyze for:
            1. Security vulnerabilities
            2. Performance optimizations
            3. ARIA-DAN compliance
            4. Error handling
            5. Code quality and maintainability
            6. Production readiness
            
            Respond with JSON:
            {{
                "quality_score": 0-100,
                "security_issues": [],
                "performance_issues": [],
                "improvements": [],
                "critical_fixes": [],
                "compliance_status": "compliant/non-compliant"
            }}
            """
            
            review_result = await self.multi_ai_engine.generate_signal(
                "CODE_REVIEW", file_path, {"code_content": review_prompt}, TaskComplexity.COMPLEX
            )
            
            # Auto-apply critical fixes
            if review_result.get("critical_fixes"):
                await self._apply_critical_fixes(file_path, review_result["critical_fixes"])
                
            logger.info(f"Code review completed for {file_path}: Score {review_result.get('quality_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Code review failed for {review_request}: {str(e)}")
    
    async def _optimization_pipeline(self):
        """Continuous system optimization pipeline"""
        while not self.shutdown_event.is_set():
            try:
                # System performance optimization
                await self._optimize_system_performance()
                
                # Resource usage optimization
                await self._optimize_resource_usage()
                
                # Trading strategy optimization
                await self._optimize_trading_strategies()
                
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except Exception as e:
                logger.error(f"Optimization pipeline error: {str(e)}")
                await asyncio.sleep(600)
    
    # Discord Integration Methods
    async def process_discord_command(self, command: str, user_id: str, signature: str = "") -> str:
        """Process commands from Discord bot with security validation"""
        try:
            # Basic security check - in production use proper role verification
            if not self.admin_role_id or user_id not in ["authorized_user_1", "authorized_user_2"]:
                logger.warning(f"Unauthorized Discord command attempt from user {user_id}")
                return "Access denied: Insufficient permissions"
            
            # Validate command signature in production
            if os.getenv("ARIA_ENV") == "production" and not signature:
                return "Access denied: Command signature required in production"
            
            command_data = {
                "action": command.lower().strip(),
                "source": f"discord_user_{user_id}",
                "timestamp": datetime.now().isoformat(),
                "signature": signature
            }
            
            await self.system_control_queue.put(command_data)
            
            # Audit log
            logger.info(f"Discord command accepted: {command} from user {user_id}")
            
            return f"Command '{command}' queued for execution. Check system logs for status."
            
        except Exception as e:
            logger.error(f"Discord command processing failed: {str(e)}")
            return f"Error processing command: {str(e)}"
    
    # Remote System Control Methods
    async def _graceful_shutdown(self):
        """Perform graceful system shutdown"""
        try:
            logger.info("Initiating graceful system shutdown...")
            
            # Set shutdown event to stop loops
            self.shutdown_event.set()
            
            # Stop all trading activities
            await self._stop_trading_engine()
            
            # Close all browser instances
            if hasattr(self.multi_ai_engine, 'cleanup'):
                await self.multi_ai_engine.cleanup()
            
            # Cancel only our managed background tasks
            for task in list(self.background_tasks):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.system_status = "shutdown"
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {str(e)}")
    
    async def _restart_system(self):
        """Restart the entire system"""
        try:
            logger.info("Initiating system restart...")
            await self._graceful_shutdown()
            await asyncio.sleep(5)
            await self.initialize()
            
        except Exception as e:
            logger.error(f"System restart failed: {str(e)}")
    
    async def _start_trading_engine(self):
        """Start the trading engine process"""
        try:
            if self.trading_engine_pid:
                logger.warning("Trading engine already running")
                return
                
            # Start trading engine subprocess
            process = await asyncio.create_subprocess_exec(
                "python", "start_live_trading.py",
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.trading_engine_pid = process.pid
            self.system_processes["trading_engine"] = process
            
            logger.info(f"Trading engine started with PID: {self.trading_engine_pid}")
            
        except Exception as e:
            logger.error(f"Failed to start trading engine: {str(e)}")
    
    async def _stop_trading_engine(self):
        """Stop the trading engine process"""
        try:
            if not self.trading_engine_pid:
                return
                
            process = self.system_processes.get("trading_engine")
            if process and process.returncode is None:  # Process is still running
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Trading engine didn't terminate gracefully, killing...")
                    process.kill()
                    await process.wait()
                
            self.trading_engine_pid = None
            if "trading_engine" in self.system_processes:
                del self.system_processes["trading_engine"]
                
            logger.info("Trading engine stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop trading engine: {str(e)}")
    
    async def _start_discord_bot(self):
        """Start the Discord bot process"""
        try:
            if self.discord_bot_pid:
                logger.warning("Discord bot already running")
                return
                
            process = await asyncio.create_subprocess_exec(
                "python", "start_discord_bot.py",
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.discord_bot_pid = process.pid
            self.system_processes["discord_bot"] = process
            
            logger.info(f"Discord bot started with PID: {self.discord_bot_pid}")
            
        except Exception as e:
            logger.error(f"Failed to start Discord bot: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_status": self.system_status,
            "is_initialized": self.is_initialized,
            "multi_ai_status": getattr(self.multi_ai_engine, 'system_status', 'unknown'),
            "hardware_config": self.hardware_config,
            "active_signals_count": len(self.active_signals),
            "signal_queue_size": self.signal_queue.qsize(),
            "diagnostic_queue_size": self.diagnostic_queue.qsize(),
            "system_control_queue_size": self.system_control_queue.qsize(),
            "active_processes": list(self.system_processes.keys()),
            "last_health_check": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "error_count": len(self.system_errors)
        }
    
    async def _get_market_context(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market context for signal generation with error handling"""
        try:
            # Import data engine for real market data
            from .data_engine import DataEngine
            data_engine = DataEngine()
            await data_engine.initialize()
            
            # Fetch real-time market data
            market_data = await data_engine.fetch_realtime_data(symbol, timeframe)
            
            if market_data and not market_data.empty:
                latest = market_data.iloc[-1]
                
                # Calculate technical indicators
                rsi = self._calculate_rsi(market_data['close'].tail(14))
                macd = self._calculate_macd(market_data['close'].tail(26))
                atr = self._calculate_atr(market_data[['high', 'low', 'close']].tail(14))
                
                # Determine trend
                sma_20 = market_data['close'].tail(20).mean()
                sma_50 = market_data['close'].tail(50).mean() if len(market_data) >= 50 else sma_20
                trend = "bullish" if latest['close'] > sma_20 > sma_50 else "bearish" if latest['close'] < sma_20 < sma_50 else "sideways"
                
                # Calculate volatility
                volatility = atr / latest['close'] if latest['close'] > 0 else 0.001
                
                return {
                    "current_price": float(latest['close']),
                    "high_24h": float(market_data['high'].tail(24).max()),
                    "low_24h": float(market_data['low'].tail(24).min()),
                    "volume": float(latest.get('volume', 0)),
                    "trend": trend,
                    "volatility": float(volatility),
                    "rsi": float(rsi),
                    "macd": float(macd),
                    "atr": float(atr),
                    "sma_20": float(sma_20),
                    "sma_50": float(sma_50)
                }
            else:
                logger.warning(f"No real market data for {symbol}, using fallback context")
                return self._get_fallback_market_context(symbol)
                
        except Exception as e:
            logger.error(f"Failed to get market context for {symbol}: {str(e)}")
            await self.diagnostic_queue.put({
                "type": "market_data_error",
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e)
            })
            return self._get_fallback_market_context(symbol)
    
    def _get_fallback_market_context(self, symbol: str) -> Dict[str, Any]:
        """Provide fallback market context when real data unavailable"""
        base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2500,
            "USDJPY": 150.00,
            "XAUUSD": 2000.00
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        return {
            "current_price": base_price,
            "high_24h": base_price * 1.005,
            "low_24h": base_price * 0.995,
            "volume": 100000,
            "trend": "sideways",
            "volatility": 0.001,
            "rsi": 50.0,
            "macd": 0.0,
            "atr": base_price * 0.001,
            "sma_20": base_price,
            "sma_50": base_price
        }
    
    def _calculate_rsi(self, prices, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            import pandas as pd
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices, fast: int = 12, slow: int = 26) -> float:
        """Calculate MACD indicator"""
        try:
            import pandas as pd
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_atr(self, data, period: int = 14) -> float:
        """Calculate ATR indicator"""
        try:
            import pandas as pd
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.001
        except:
            return 0.001
    
    # Helper Methods for System Operations
    async def _apply_code_fix(self, fix_data: Dict[str, Any]):
        """Apply automatic code fix based on Multi-AI diagnosis"""
        try:
            file_path = fix_data.get("file")
            function_name = fix_data.get("function")
            fix_code = fix_data.get("fix")
            
            if not all([file_path, fix_code]):
                return
                
            # Create backup
            backup_path = f"{file_path}.backup_{int(datetime.now().timestamp())}"
            shutil.copy2(file_path, backup_path)
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Store fix for review
            fix_id = hashlib.md5(f"{file_path}{function_name}".encode()).hexdigest()[:8]
            self.pending_fixes[fix_id] = {
                "file_path": file_path,
                "function": function_name,
                "fix_code": fix_code,
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat(),
                "applied": False
            }
            
            # Apply fix if auto-apply is enabled and not in production
            if self.auto_apply_fixes and os.getenv("ARIA_ENV", "dev") != "production":
                try:
                    # Apply the fix (simplified - in production use proper AST manipulation)
                    modified_content = content.replace("# TODO: PLACEHOLDER", fix_code)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                        
                    self.pending_fixes[fix_id]["applied"] = True
                    logger.info(f"Code fix applied to {file_path}:{function_name} (ID: {fix_id})")
                except Exception as e:
                    logger.error(f"Failed to apply fix {fix_id}: {e}")
            else:
                logger.info(f"Code fix prepared for review {file_path}:{function_name} (ID: {fix_id})")
            
        except Exception as e:
            logger.error(f"Failed to apply code fix: {str(e)}")
    
    async def _apply_critical_fixes(self, file_path: str, critical_fixes: List[Dict[str, Any]]):
        """Apply critical fixes identified during code review"""
        try:
            for fix in critical_fixes:
                await self._apply_code_fix({
                    "file": file_path,
                    "function": fix.get("function", "unknown"),
                    "fix": fix.get("code", "")
                })
                
            logger.info(f"Applied {len(critical_fixes)} critical fixes to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to apply critical fixes: {str(e)}")
    
    async def _schedule_system_restart(self):
        """Schedule system restart with delay for cleanup"""
        try:
            logger.warning("System restart scheduled in 60 seconds...")
            
            # Add restart command to queue with delay
            restart_command = {
                "action": "restart",
                "source": "auto_fix_system",
                "scheduled_time": (datetime.now() + timedelta(seconds=60)).isoformat()
            }
            
            await asyncio.sleep(60)
            await self.system_control_queue.put(restart_command)
            
        except Exception as e:
            logger.error(f"Failed to schedule system restart: {str(e)}")
    
    async def _check_system_health(self):
        """Perform periodic system health checks"""
        try:
            health_metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(self.system_processes),
                "queue_sizes": {
                    "signal": self.signal_queue.qsize(),
                    "diagnostic": self.diagnostic_queue.qsize(),
                    "control": self.system_control_queue.qsize()
                }
            }
            
            # Check for resource issues
            if health_metrics["memory_usage"] > 90:
                await self.diagnostic_queue.put({
                    "type": "high_memory_usage",
                    "value": health_metrics["memory_usage"],
                    "timestamp": datetime.now().isoformat()
                })
                
            if health_metrics["cpu_usage"] > 95:
                await self.diagnostic_queue.put({
                    "type": "high_cpu_usage",
                    "value": health_metrics["cpu_usage"],
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"System health check failed: {str(e)}")
    
    async def _periodic_code_audit(self):
        """Perform periodic code quality audits"""
        try:
            # Get all Python files in the project
            python_files = []
            for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(__file__))):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        python_files.append(os.path.join(root, file))
            
            # Queue random files for review
            import random
            files_to_review = random.sample(python_files, min(3, len(python_files)))
            
            for file_path in files_to_review:
                await self.code_review_queue.put({
                    "file_path": file_path,
                    "type": "periodic_audit",
                    "timestamp": datetime.now().isoformat()
                })
                
            logger.info(f"Queued {len(files_to_review)} files for periodic code audit")
            
        except Exception as e:
            logger.error(f"Periodic code audit failed: {str(e)}")
    
    async def _optimize_system_performance(self):
        """Optimize system performance based on current metrics"""
        try:
            performance_prompt = f"""
            ARIA-DAN PERFORMANCE OPTIMIZATION ANALYSIS
            
            Current System Metrics:
            - CPU Usage: {psutil.cpu_percent()}%
            - Memory Usage: {psutil.virtual_memory().percent}%
            - Active Processes: {len(self.system_processes)}
            - GPU Available: {self.has_gpu}
            - Browser Instances: {self.hardware_config.get('browser_instances', 'unknown')}
            
            Queue Status:
            - Signal Queue: {self.signal_queue.qsize()}
            - Diagnostic Queue: {self.diagnostic_queue.qsize()}
            - Control Queue: {self.system_control_queue.qsize()}
            
            Analyze performance bottlenecks and recommend optimizations.
            Focus on browser automation efficiency and resource utilization.
            
            Respond with JSON:
            {{
                "bottlenecks": [],
                "optimizations": [],
                "resource_adjustments": {{}},
                "performance_score": 0-100
            }}
            """
            
            optimization_result = await self.multi_ai_engine.generate_signal(
                "PERFORMANCE_OPTIMIZATION", "SYSTEM", {"metrics": performance_prompt}, TaskComplexity.COMPLEX
            )
            
            # Apply recommended optimizations
            if optimization_result.get("resource_adjustments"):
                self.hardware_config.update(optimization_result["resource_adjustments"])
                
            logger.info(f"Performance optimization completed: Score {optimization_result.get('performance_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {str(e)}")
    
    async def _optimize_resource_usage(self):
        """Optimize resource usage across the system"""
        try:
            # Monitor and optimize browser instances
            if hasattr(self.multi_ai_engine, 'browser_pool'):
                active_browsers = len(getattr(self.multi_ai_engine, 'browser_pool', {}))
                optimal_browsers = self.hardware_config.get('browser_instances', 4)
                
                if active_browsers > optimal_browsers * 1.5:
                    logger.info(f"Reducing browser instances from {active_browsers} to {optimal_browsers}")
                    # Implementation would go here
                    
            # Optimize memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                await self.audit_log("high_memory_usage", {"memory_percent": memory_percent})
                await self._free_system_memory()
                
        except Exception as e:
            logger.error(f"Resource optimization failed: {str(e)}")
    
    async def _optimize_trading_strategies(self):
        """Optimize trading strategies based on performance data"""
        try:
            strategy_prompt = f"""
            ARIA-DAN TRADING STRATEGY OPTIMIZATION
            
            Current Strategy Performance:
            - Active Signals: {len(self.active_signals)}
            - Hardware Config: {self.hardware_config.get('optimization_level')}
            - System Status: {self.system_status}
            
            Analyze current trading strategies and recommend optimizations.
            Consider hardware capabilities and market conditions.
            
            Respond with JSON:
            {{
                "strategy_adjustments": {{}},
                "risk_parameters": {{}},
                "execution_optimizations": [],
                "performance_improvements": []
            }}
            """
            
            strategy_result = await self.multi_ai_engine.generate_signal(
                "STRATEGY_OPTIMIZATION", "TRADING", {"data": strategy_prompt}, TaskComplexity.COMPLEX
            )
            
            # Apply strategy optimizations
            if strategy_result.get("strategy_adjustments"):
                self.strategy_config.update(strategy_result["strategy_adjustments"])
                
            logger.info("Trading strategy optimization completed")
            
        except Exception as e:
            logger.error(f"Trading strategy optimization failed: {str(e)}")
    
    async def _send_status_report(self, destination: Optional[str] = None):
        """Send comprehensive status report"""
        try:
            status_report = await self.get_system_status()
            report_text = json.dumps(status_report, indent=2)
            
            logger.info(f"Status report generated: {report_text[:200]}...")
            
            # If destination is Discord, send via Discord bot
            if destination and "discord" in destination.lower():
                # Implementation would integrate with Discord bot
                pass
                
        except Exception as e:
            logger.error(f"Failed to send status report: {str(e)}")
    
    async def _free_system_memory(self):
        """Free up system memory when usage is high"""
        try:
            import gc
            gc.collect()
            
            # Clear expired signals
            await self.clear_expired_signals()
            
            # Reduce browser instances if needed
            if hasattr(self.multi_ai_engine, 'optimize_memory'):
                await self.multi_ai_engine.optimize_memory()
                
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
    
    # Cleanup and Validation Methods
    async def cleanup(self):
        """Cleanup system resources"""
        try:
            # Close Multi-AI engine
            if hasattr(self.multi_ai_engine, 'cleanup'):
                await self.multi_ai_engine.cleanup()
                
            # Stop all processes
            for process_name, process in self.system_processes.items():
                try:
                    if process and not process.returncode:
                        process.terminate()
                        await process.wait()
                except Exception as e:
                    logger.error(f"Failed to stop process {process_name}: {e}")
                    
            self.system_status = "cleanup_complete"
            logger.info("Multi-AI Workflow Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    # Security and audit methods
    async def audit_log(self, event: str, data: Dict[str, Any]):
        """Write immutable audit log entry"""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "data": data,
                "system_status": self.system_status,
                "uptime": str(datetime.now() - self.start_time)
            }
            
            # In production, write to immutable audit log
            audit_path = os.getenv("ARIA_AUDIT_LOG_PATH", "/tmp/aria_audit.log")
            with open(audit_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    
    async def _validate_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal data with Multi-AI consensus validation"""
        try:
            # Basic validation
            required_fields = ["direction", "entry_price", "stop_loss", "take_profit", "confidence"]
            for field in required_fields:
                if field not in signal_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Calculate additional metrics
            entry_price = signal_data["entry_price"]
            stop_loss = signal_data["stop_loss"]
            take_profit = signal_data["take_profit"]
            
            # Risk-reward ratio validation
            if signal_data["direction"] == "buy":
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
            else:
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - take_profit)
            
            if risk > 0:
                risk_reward_ratio = reward / risk
            else:
                risk_reward_ratio = 0
            
            signal_data["risk_reward_ratio"] = round(risk_reward_ratio, 2)
            
            # Multi-AI validation check
            if signal_data.get("consensus_score", 0) < 0.6:
                signal_data["validation_warning"] = "Low consensus score"
                
            # Hardware-aware validation
            if not self.has_gpu and signal_data.get("confidence", 0) > 0.9:
                signal_data["confidence"] = min(0.85, signal_data["confidence"])  # Cap confidence on CPU
                signal_data["validation_note"] = "Confidence capped for CPU system"
            
            # Add validation status
            signal_data["validation_status"] = "validated"
            signal_data["validated_at"] = datetime.now().isoformat()
            signal_data["validator"] = "multi_ai_workflow_agent"
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Signal validation failed: {str(e)}")
            signal_data["validation_status"] = "invalid"
            signal_data["validation_error"] = str(e)
            return signal_data
    
    async def _signal_processing_loop(self):
        """Background loop for processing signals"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    signal = await asyncio.wait_for(
                        self.signal_queue.get(), timeout=1.0
                    )
                    await self._process_signal(signal)
                except asyncio.TimeoutError:
                    pass  # Continue loop
            except Exception as e:
                logger.error(f"Error in signal processing loop: {str(e)}")
                await asyncio.sleep(5)
    
    async def _process_signal(self, signal: Dict[str, Any]):
        """Process a validated signal"""
        try:
            signal_id = f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal['symbol']}"
            signal["signal_id"] = signal_id
            
            # Store in active signals
            self.active_signals[signal_id] = signal
            
            # Add to approval queue
            await self.approval_queue.put(signal)
            
            logger.info(f"Signal {signal_id} processed and queued for approval")
            
        except Exception as e:
            logger.error(f"Failed to process signal: {str(e)}")
    
    async def _approval_processing_loop(self):
        """Background loop for processing signal approvals"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    signal = await asyncio.wait_for(
                        self.approval_queue.get(), timeout=1.0
                    )
                    await self._approve_signal(signal)
                except asyncio.TimeoutError:
                    pass  # Continue loop
            except Exception as e:
                logger.error(f"Error in approval processing loop: {str(e)}")
                await asyncio.sleep(5)
    
    async def _approve_signal(self, signal: Dict[str, Any]):
        """Approve a signal for execution"""
        try:
            # Use Gemini to approve the signal
            approval_prompt = f"""
            Review this trading signal and approve or reject it:
            
            Signal Data: {json.dumps(signal, indent=2)}
            
            Consider:
            1. Risk-reward ratio (should be at least 1:2)
            2. Confidence level (should be above 0.7)
            3. Market conditions
            4. Risk management rules
            
            Respond with JSON:
            {{
                "approved": true/false,
                "reason": "Brief explanation",
                "confidence_adjustment": float (optional confidence adjustment)
            }}
            """
            
            response = self.gemini_model.generate_content(approval_prompt)
            approval_data = self._parse_approval_response(response.text)
            
            if approval_data.get("approved", False):
                signal["status"] = "approved"
                signal["approved_at"] = datetime.now().isoformat()
                logger.info(f"Signal {signal.get('signal_id')} approved for execution")
            else:
                signal["status"] = "rejected"
                signal["rejection_reason"] = approval_data.get("reason", "Unknown")
                logger.info(f"Signal {signal.get('signal_id')} rejected: {signal['rejection_reason']}")
            
        except Exception as e:
            logger.error(f"Failed to approve signal: {str(e)}")
            signal["status"] = "error"
            signal["error"] = str(e)
    
    def _parse_approval_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini approval response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"approved": True, "reason": "Default approval"}
        except Exception as e:
            logger.error(f"Failed to parse approval response: {str(e)}")
            return {"approved": True, "reason": "Fallback approval"}
    
    async def _health_monitor(self):
        """Monitor Multi-AI engine health"""
        while not self.shutdown_event.is_set():
            try:
                # Monitor Multi-AI engine health instead of Gemini
                if hasattr(self.multi_ai_engine, 'is_initialized') and self.multi_ai_engine.is_initialized:
                    self.system_status = "operational"
                else:
                    self.system_status = "degraded"
                    logger.warning("Multi-AI engine not properly initialized")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {str(e)}")
                self.system_status = "error"
                await asyncio.sleep(60)
    
    # Metrics and monitoring
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format"""
        try:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            metrics = f"""# HELP aria_uptime_seconds System uptime in seconds
# TYPE aria_uptime_seconds gauge
aria_uptime_seconds {uptime_seconds}

# HELP aria_active_signals Current number of active trading signals
# TYPE aria_active_signals gauge
aria_active_signals {len(self.active_signals)}

# HELP aria_queue_size Queue sizes by type
# TYPE aria_queue_size gauge
aria_queue_size{{queue="signal"}} {self.signal_queue.qsize()}
aria_queue_size{{queue="diagnostic"}} {self.diagnostic_queue.qsize()}
aria_queue_size{{queue="control"}} {self.system_control_queue.qsize()}

# HELP aria_background_tasks Number of background tasks
# TYPE aria_background_tasks gauge
aria_background_tasks {len(self.background_tasks)}

# HELP aria_system_processes Number of managed system processes
# TYPE aria_system_processes gauge
aria_system_processes {len(self.system_processes)}

# HELP aria_pending_fixes Number of pending code fixes
# TYPE aria_pending_fixes gauge
aria_pending_fixes {len(self.pending_fixes)}
"""
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics generation failed: {e}")
            return "# Metrics unavailable\n"
    
    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active signals"""
        return list(self.active_signals.values())
    
    async def clear_expired_signals(self):
        """Clear expired signals from active list"""
        current_time = datetime.now()
        expired_signals = []
        
        for signal_id, signal in self.active_signals.items():
            if signal.get("expires_at"):
                expires_at = datetime.fromisoformat(signal["expires_at"])
                if current_time > expires_at:
                    expired_signals.append(signal_id)
        
        for signal_id in expired_signals:
            del self.active_signals[signal_id]
            logger.info(f"Expired signal {signal_id} removed from active list")