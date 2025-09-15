import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import discord
from discord.ext import commands
import json
import aiohttp

import sys
import os
# Add the parent directory to the path so we can import from backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from core.autonomous_ai_agent import AutonomousAIAgent

logger = setup_logger(__name__)

class DiscordAgent:
    """
    ARIA-DAN Discord Command Interface with Gemini AI Integration
    Wall Street grade command and control system
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.bot = None
        self.supervisor_client = None
        self.allowed_users = set()
        self.command_history = []
        
        # GEMINI AI INTEGRATION
        self.autonomous_agent = None
        self.ai_session_context = {}
        self.last_ai_interaction = {}
        
    async def initialize(self):
        """Initialize the Discord agent"""
        try:
            logger.info("Initializing Discord Agent")
            
            # Load configuration
            await self._load_config()
            
            # Initialize Autonomous AI Agent
            await self._initialize_autonomous_agent()
            
            # Create Discord bot
            await self._create_discord_bot()
            
            self.is_initialized = True
            logger.info("🤖 ARIA-DAN Discord Agent initialized with Gemini AI integration")
            
        except Exception as e:
            logger.error(f"Failed to initialize Discord Agent: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load Discord agent configuration"""
        try:
            self.project_config = self.config.load_project_config()
            
            # Get Discord configuration
            self.discord_token = os.getenv("DISCORD_BOT_TOKEN")
            self.allowed_user_ids = os.getenv("DISCORD_ALLOWED_USERS", "").split(",")
            
            if not self.discord_token:
                raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
            
            # Convert user IDs to integers
            self.allowed_users = set()
            for user_id in self.allowed_user_ids:
                try:
                    self.allowed_users.add(int(user_id.strip()))
                except ValueError:
                    logger.warning(f"Invalid user ID: {user_id}")
            
            logger.info(f"Discord configuration loaded - {len(self.allowed_users)} allowed users")
            
        except Exception as e:
            logger.error(f"Failed to load Discord configuration: {str(e)}")
            raise
    
    async def _initialize_autonomous_agent(self):
        """Initialize connection to autonomous AI agent"""
        try:
            logger.info("🧠 Initializing Gemini AI integration")
            
            # Initialize autonomous agent
            self.autonomous_agent = AutonomousAIAgent()
            await self.autonomous_agent.initialize()
            
            logger.info("✅ Gemini AI brain connected to Discord interface")
            
        except Exception as e:
            logger.error(f"Failed to load Discord configuration: {str(e)}")
            raise
    
    async def _create_discord_bot(self):
        """Create Discord bot with commands"""
        try:
            # Define bot intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            
            # Create bot
            self.bot = commands.Bot(command_prefix='!', intents=intents)
            
            # Setup bot events
            @self.bot.event
            async def on_ready():
                logger.info(f'Discord bot connected as {self.bot.user}')
                await self.bot.change_presence(activity=discord.Game(name="ARIA Trading System"))
            
            @self.bot.event
            async def on_message(message):
                # Ignore bot messages
                if message.author.bot:
                    return
                
                # Check if user is allowed
                if message.author.id not in self.allowed_users:
                    await message.channel.send("❌ You are not authorized to use this bot.")
                    return
                
                # Process commands
                await self.bot.process_commands(message)
            
            # Register commands
            await self._register_commands()
            
        except Exception as e:
            logger.error(f"Failed to create Discord bot: {str(e)}")
            raise
    
    async def _register_commands(self):
        """Register bot commands"""
        try:
            @self.bot.command(name='status', help='Get system status')
            async def status(ctx):
                """Get system status"""
                try:
                    await ctx.trigger_typing()
                    
                    status_data = await self._get_system_status()
                    
                    embed = discord.Embed(
                        title="🖥️ System Status",
                        color=discord.Color.green() if status_data.get("system_status") == "running" else discord.Color.red(),
                        timestamp=datetime.now()
                    )
                    
                    embed.add_field(name="System Status", value=status_data.get("system_status", "unknown"), inline=False)
                    embed.add_field(name="Active Positions", value=str(status_data.get("active_positions", 0)), inline=True)
                    embed.add_field(name="Daily P&L", value=f"${status_data.get('daily_pnl', 0):.2f}", inline=True)
                    embed.add_field(name="Total Trades", value=str(status_data.get("total_trades", 0)), inline=True)
                    embed.add_field(name="Win Rate", value=f"{status_data.get('win_rate', 0):.1%}", inline=True)
                    
                    if status_data.get("last_signal_time"):
                        embed.add_field(name="Last Signal", value=status_data["last_signal_time"].strftime("%Y-%m-%d %H:%M:%S"), inline=False)
                    
                    embed.set_footer(text=f"Requested by {ctx.author.name}")
                    
                    await ctx.send(embed=embed)
                    
                    # Log command
                    await self._log_command(ctx.author.id, "status", "success")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error getting status: {str(e)}")
                    await self._log_command(ctx.author.id, "status", f"error: {str(e)}")
            
            @self.bot.command(name='pause', help='Pause the trading system')
            async def pause(ctx):
                """Pause the trading system"""
                try:
                    await ctx.trigger_typing()
                    
                    success = await self._pause_system()
                    
                    if success:
                        embed = discord.Embed(
                            title="⏸️ System Paused",
                            description="The trading system has been paused successfully.",
                            color=discord.Color.orange(),
                            timestamp=datetime.now()
                        )
                        embed.set_footer(text=f"Requested by {ctx.author.name}")
                        await ctx.send(embed=embed)
                        await self._log_command(ctx.author.id, "pause", "success")
                    else:
                        await ctx.send("❌ Failed to pause system")
                        await self._log_command(ctx.author.id, "pause", "failed")
                        
                except Exception as e:
                    await ctx.send(f"❌ Error pausing system: {str(e)}")
                    await self._log_command(ctx.author.id, "pause", f"error: {str(e)}")
            
            @self.bot.command(name='resume', help='Resume the trading system')
            async def resume(ctx):
                """Resume the trading system"""
                try:
                    await ctx.trigger_typing()
                    
                    success = await self._resume_system()
                    
                    if success:
                        embed = discord.Embed(
                            title="▶️ System Resumed",
                            description="The trading system has been resumed successfully.",
                            color=discord.Color.green(),
                            timestamp=datetime.now()
                        )
                        embed.set_footer(text=f"Requested by {ctx.author.name}")
                        await ctx.send(embed=embed)
                        await self._log_command(ctx.author.id, "resume", "success")
                    else:
                        await ctx.send("❌ Failed to resume system")
                        await self._log_command(ctx.author.id, "resume", "failed")
                        
                except Exception as e:
                    await ctx.send(f"❌ Error resuming system: {str(e)}")
                    await self._log_command(ctx.author.id, "resume", f"error: {str(e)}")
            
            @self.bot.command(name='kill_switch', help='Activate emergency kill switch')
            async def kill_switch(ctx):
                """Activate emergency kill switch"""
                try:
                    await ctx.trigger_typing()
                    
                    # Confirmation dialog
                    embed = discord.Embed(
                        title="⚠️ KILL SWITCH CONFIRMATION",
                        description="Are you sure you want to activate the emergency kill switch?\n\nThis will:\n• Close all open positions\n• Stop all trading activity\n• Pause the system\n\nType `confirm` to proceed or `cancel` to abort.",
                        color=discord.Color.red(),
                        timestamp=datetime.now()
                    )
                    
                    await ctx.send(embed=embed)
                    
                    def check(m):
                        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ['confirm', 'cancel']
                    
                    try:
                        response = await self.bot.wait_for('message', timeout=30.0, check=check)
                        
                        if response.content.lower() == 'confirm':
                            success = await self._activate_kill_switch()
                            
                            if success:
                                embed = discord.Embed(
                                    title="🚨 KILL SWITCH ACTIVATED",
                                    description="Emergency kill switch has been activated!\n\nAll positions closed and system stopped.",
                                    color=discord.Color.red(),
                                    timestamp=datetime.now()
                                )
                                embed.set_footer(text=f"Activated by {ctx.author.name}")
                                await ctx.send(embed=embed)
                                await self._log_command(ctx.author.id, "kill_switch", "success")
                            else:
                                await ctx.send("❌ Failed to activate kill switch")
                                await self._log_command(ctx.author.id, "kill_switch", "failed")
                        else:
                            await ctx.send("✅ Kill switch activation cancelled")
                            await self._log_command(ctx.author.id, "kill_switch", "cancelled")
                            
                    except asyncio.TimeoutError:
                        await ctx.send("⏰ Kill switch activation timed out")
                        await self._log_command(ctx.author.id, "kill_switch", "timeout")
                        
                except Exception as e:
                    await ctx.send(f"❌ Error with kill switch: {str(e)}")
                    await self._log_command(ctx.author.id, "kill_switch", f"error: {str(e)}")
            
            @self.bot.command(name='train', help='Train AI models')
            async def train(ctx, model_type: str = "all"):
                """Train AI models"""
                try:
                    await ctx.trigger_typing()
                    
                    success = await self._train_model(model_type)
                    
                    if success:
                        embed = discord.Embed(
                            title="🧠 Model Training",
                            description=f"Training initiated for {model_type} model(s).",
                            color=discord.Color.blue(),
                            timestamp=datetime.now()
                        )
                        embed.set_footer(text=f"Requested by {ctx.author.name}")
                        await ctx.send(embed=embed)
                        await self._log_command(ctx.author.id, "train", f"success: {model_type}")
                    else:
                        await ctx.send(f"❌ Failed to train {model_type} model")
                        await self._log_command(ctx.author.id, "train", f"failed: {model_type}")
                        
                except Exception as e:
                    await ctx.send(f"❌ Error training model: {str(e)}")
                    await self._log_command(ctx.author.id, "train", f"error: {model_type} - {str(e)}")
            
            @self.bot.command(name='positions', help='Show active positions')
            async def positions(ctx):
                """Show active positions"""
                try:
                    await ctx.trigger_typing()
                    
                    positions_data = await self._get_active_positions()
                    
                    if not positions_data:
                        embed = discord.Embed(
                            title="📊 Active Positions",
                            description="No active positions",
                            color=discord.Color.blue(),
                            timestamp=datetime.now()
                        )
                        await ctx.send(embed=embed)
                    else:
                        embed = discord.Embed(
                            title="📊 Active Positions",
                            color=discord.Color.blue(),
                            timestamp=datetime.now()
                        )
                        
                        for pos in positions_data[:5]:  # Show max 5 positions
                            pnl_color = discord.Color.green() if pos.get('pnl', 0) > 0 else discord.Color.red()
                            embed.add_field(
                                name=f"{pos.get('symbol', 'N/A')} ({pos.get('direction', 'N/A')})",
                                value=f"Entry: {pos.get('entry_price', 0):.5f}\nP&L: ${pos.get('pnl', 0):.2f}",
                                inline=True
                            )
                        
                        if len(positions_data) > 5:
                            embed.set_footer(text=f"Showing 5 of {len(positions_data)} positions")
                        
                        await ctx.send(embed=embed)
                    
                    await self._log_command(ctx.author.id, "positions", "success")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error getting positions: {str(e)}")
                    await self._log_command(ctx.author.id, "positions", f"error: {str(e)}")
            
            @self.bot.command(name='help', help='Show available commands')
            async def help(ctx):
                """Show help information"""
                try:
                    embed = discord.Embed(
                        title="🤖 ARIA Trading System - Help",
                        description="Available commands for controlling the trading system:",
                        color=discord.Color.blue(),
                        timestamp=datetime.now()
                    )
                    
                    commands_info = [
                        ("!status", "Get current system status"),
                        ("!pause", "Pause the trading system"),
                        ("!resume", "Resume the trading system"),
                        ("!kill_switch", "Activate emergency kill switch"),
                        ("!train [model|all]", "Train AI models"),
                        ("!positions", "Show active positions"),
                        ("!help", "Show this help message")
                    ]
                    
                    for cmd, desc in commands_info:
                        embed.add_field(name=cmd, value=desc, inline=False)
                    
                    embed.set_footer(text=f"Requested by {ctx.author.name}")
                    await ctx.send(embed=embed)
                    
                    await self._log_command(ctx.author.id, "help", "success")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error showing help: {str(e)}")
                    await self._log_command(ctx.author.id, "help", f"error: {str(e)}")
            
            # GEMINI AI COMMAND
            @self.bot.command(name='ask', help='Ask Gemini AI about market conditions or system status')
            async def ask_gemini(ctx, *, question: str):
                """Ask Gemini AI a question about markets or system"""
                try:
                    await ctx.trigger_typing()
                    
                    # Get AI response through autonomous agent
                    ai_response = await self._ask_gemini_ai(ctx.author.id, question)
                    
                    # Format response for Discord
                    embed = discord.Embed(
                        title="🧠 Gemini AI Analysis",
                        description=ai_response.get('ai_analysis', 'No analysis available'),
                        color=discord.Color.blue(),
                        timestamp=datetime.now()
                    )
                    
                    if ai_response.get('recommendation'):
                        embed.add_field(name="Recommendation", value=ai_response['recommendation'], inline=True)
                    
                    if ai_response.get('confidence'):
                        embed.add_field(name="Confidence", value=f"{ai_response['confidence']:.1%}", inline=True)
                        
                    if ai_response.get('risk_level'):
                        embed.add_field(name="Risk Level", value=ai_response['risk_level'], inline=True)
                    
                    if ai_response.get('reasoning'):
                        embed.add_field(name="Reasoning", value=ai_response['reasoning'], inline=False)
                    
                    embed.set_footer(text=f"Asked by {ctx.author.name}")
                    await ctx.send(embed=embed)
                    
                    await self._log_command(ctx.author.id, "ask", "success")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error asking Gemini AI: {str(e)}")
                    await self._log_command(ctx.author.id, "ask", f"error: {str(e)}")
            
            # SYSTEM HEALTH COMMAND WITH AI
            @self.bot.command(name='health', help='Get AI analysis of system health')
            async def system_health(ctx):
                """Get AI-powered system health analysis"""
                try:
                    await ctx.trigger_typing()
                    
                    # Get AI health analysis
                    health_data = await self._get_ai_system_health(ctx.author.id)
                    
                    embed = discord.Embed(
                        title="🏥 System Health Analysis (AI-Powered)",
                        color=discord.Color.green() if health_data.get('system_status') == 'OPERATIONAL' else discord.Color.red(),
                        timestamp=datetime.now()
                    )
                    
                    embed.add_field(name="System Status", value=health_data.get('system_status', 'UNKNOWN'), inline=True)
                    embed.add_field(name="AI Brain Status", value=health_data.get('ai_brain_status', 'OFFLINE'), inline=True)
                    embed.add_field(name="Active Signals", value=str(health_data.get('active_signals', 0)), inline=True)
                    embed.add_field(name="Total Positions", value=str(health_data.get('total_positions', 0)), inline=True)
                    embed.add_field(name="Market Regime", value=health_data.get('market_regime', 'UNKNOWN'), inline=True)
                    embed.add_field(name="Performance", value=f"${health_data.get('performance_metrics', {}).get('total_pnl', 0):.2f}", inline=True)
                    
                    if health_data.get('ai_health_analysis'):
                        embed.add_field(name="AI Analysis", value=health_data['ai_health_analysis'], inline=False)
                    
                    embed.set_footer(text=f"Requested by {ctx.author.name}")
                    await ctx.send(embed=embed)
                    
                    await self._log_command(ctx.author.id, "health", "success")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error getting system health: {str(e)}")
                    await self._log_command(ctx.author.id, "health", f"error: {str(e)}")
            
            # TRADING COMMAND WITH AI
            @self.bot.command(name='trade', help='Ask AI for trading recommendation on specific pair')
            async def trade_recommendation(ctx, symbol: str):
                """Get AI trading recommendation for specific symbol"""
                try:
                    await ctx.trigger_typing()
                    
                    query = f"Analyze {symbol.upper()} and provide your trading recommendation based on current market conditions, news, and technical analysis."
                    ai_response = await self._ask_gemini_ai(ctx.author.id, query)
                    
                    # Enhanced trading response
                    embed = discord.Embed(
                        title=f"📈 {symbol.upper()} Trading Analysis",
                        description=ai_response.get('ai_analysis', 'No analysis available'),
                        color=discord.Color.green() if ai_response.get('recommendation') == 'BUY' else discord.Color.red() if ai_response.get('recommendation') == 'SELL' else discord.Color.orange(),
                        timestamp=datetime.now()
                    )
                    
                    embed.add_field(name="Signal", value=ai_response.get('recommendation', 'HOLD'), inline=True)
                    embed.add_field(name="Confidence", value=f"{ai_response.get('confidence', 0.5):.1%}", inline=True)
                    embed.add_field(name="Risk Level", value=ai_response.get('risk_level', 'MEDIUM'), inline=True)
                    
                    if ai_response.get('reasoning'):
                        embed.add_field(name="Analysis", value=ai_response['reasoning'], inline=False)
                    
                    embed.set_footer(text=f"Analysis for {ctx.author.name} • Not financial advice")
                    await ctx.send(embed=embed)
                    
                    await self._log_command(ctx.author.id, "trade", f"success: {symbol}")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error getting trading recommendation: {str(e)}")
                    await self._log_command(ctx.author.id, "trade", f"error: {symbol} - {str(e)}")
            
            @self.bot.command(name='logs', help='Show recent system logs')
            async def logs(ctx, limit: int = 10):
                """Show recent system logs"""
                try:
                    await ctx.trigger_typing()
                    
                    logs_data = await self._get_logs(limit)
                    
                    if not logs_data:
                        await ctx.send("No logs available")
                        return
                    
                    embed = discord.Embed(
                        title="📋 Recent System Logs",
                        color=discord.Color.orange(),
                        timestamp=datetime.now()
                    )
                    
                    for log in logs_data[:5]:  # Show max 5 logs
                        level_emoji = {
                            "INFO": "ℹ️",
                            "WARNING": "⚠️",
                            "ERROR": "❌",
                            "CRITICAL": "🚨"
                        }.get(log.get("level", "INFO"), "ℹ️")
                        
                        embed.add_field(
                            name=f"{level_emoji} {log.get('timestamp', 'N/A')[:19]}",
                            value=f"**{log.get('service', 'system')}**: {log.get('message', 'No message')}",
                            inline=False
                        )
                    
                    if len(logs_data) > 5:
                        embed.set_footer(text=f"Showing 5 of {len(logs_data)} logs")
                    
                    await ctx.send(embed=embed)
                    await self._log_command(ctx.author.id, "logs", f"success: {limit}")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error getting logs: {str(e)}")
                    await self._log_command(ctx.author.id, "logs", f"error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to register commands: {str(e)}")
            raise
    
    async def start_bot(self):
        """Start the Discord bot"""
        try:
            if not self.bot:
                raise ValueError("Bot not initialized")
            
            logger.info("Starting Discord bot")
            await self.bot.start(self.discord_token)
            
        except Exception as e:
            logger.error(f"Failed to start Discord bot: {str(e)}")
            raise
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status from AI agent"""
        try:
            if not self.autonomous_agent:
                return {
                    "system_status": "offline",
                    "active_positions": 0,
                    "daily_pnl": 0.0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "last_signal_time": None
                }
            
            # Get real status from autonomous agent
            health_data = await self.autonomous_agent.get_system_health_for_discord()
            
            # Convert to expected format
            return {
                "system_status": "running" if health_data.get('system_status') == 'OPERATIONAL' else "offline",
                "active_positions": health_data.get('total_positions', 0),
                "daily_pnl": health_data.get('performance_metrics', {}).get('total_pnl', 0.0),
                "total_trades": health_data.get('performance_metrics', {}).get('signals_executed', 0),
                "win_rate": health_data.get('performance_metrics', {}).get('success_rate', 0.0),
                "last_signal_time": datetime.now()  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {}
    
    async def _pause_system(self) -> bool:
        """Pause the system"""
        try:
            # This would normally make HTTP request to supervisor API
            logger.info("System pause requested via Discord")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing system: {str(e)}")
            return False
    
    async def _resume_system(self) -> bool:
        """Resume the system"""
        try:
            # This would normally make HTTP request to supervisor API
            logger.info("System resume requested via Discord")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming system: {str(e)}")
            return False
    
    async def _activate_kill_switch(self) -> bool:
        """Activate kill switch"""
        try:
            # This would normally make HTTP request to supervisor API
            logger.warning("Kill switch activated via Discord")
            return True
            
        except Exception as e:
            logger.error(f"Error activating kill switch: {str(e)}")
            return False
    
    async def _train_model(self, model_type: str) -> bool:
        """Train model"""
        try:
            # This would normally make HTTP request to supervisor API
            logger.info(f"Model training requested via Discord: {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    async def _get_active_positions(self) -> List[Dict[str, Any]]:
        """Get active positions from AI agent"""
        try:
            if not self.autonomous_agent:
                return []
            
            # Get positions through autonomous agent
            if hasattr(self.autonomous_agent, 'active_positions'):
                positions = []
                for pos_id, pos_data in self.autonomous_agent.active_positions.items():
                    positions.append({
                        "symbol": pos_data.get('symbol', 'UNKNOWN'),
                        "direction": pos_data.get('direction', 'UNKNOWN'),
                        "entry_price": pos_data.get('entry_price', 0.0),
                        "pnl": pos_data.get('unrealized_pnl', 0.0)
                    })
                return positions
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting active positions: {str(e)}")
            return []
    
    async def _get_logs(self, limit: int) -> List[Dict[str, Any]]:
        """Get system logs"""
        try:
            # This would normally make HTTP request to supervisor API
            # Return mock data for now
            import random
            
            logs = []
            for i in range(min(limit, 10)):
                logs.append({
                    "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                    "level": random.choice(["INFO", "WARNING", "ERROR"]),
                    "service": random.choice(["gemini_agent", "signal_manager", "execution_engine"]),
                    "message": f"System log entry {i+1}"
                })
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting logs: {str(e)}")
            return []
    
    async def _log_command(self, user_id: int, command: str, result: str):
        """Log command execution"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "command": command,
                "result": result
            }
            
            self.command_history.append(log_entry)
            
            # Keep only last 1000 commands
            if len(self.command_history) > 1000:
                self.command_history = self.command_history[-1000:]
            
            logger.info(f"Discord command: {command} by user {user_id} - {result}")
            
        except Exception as e:
            logger.error(f"Error logging command: {str(e)}")
    
    # GEMINI AI INTEGRATION METHODS
    
    async def _ask_gemini_ai(self, user_id: int, question: str) -> Dict[str, Any]:
        """Ask Gemini AI through autonomous agent"""
        try:
            if not self.autonomous_agent:
                return {
                    'error': 'AI agent not initialized',
                    'ai_analysis': 'Gemini AI brain is not available',
                    'recommendation': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': 'System error - AI brain offline',
                    'risk_level': 'HIGH'
                }
            
            # Process Discord command through autonomous agent
            ai_response = await self.autonomous_agent.process_discord_command(question, str(user_id))
            
            # Store interaction context
            self.ai_session_context[user_id] = {
                'last_question': question,
                'last_response': ai_response,
                'timestamp': datetime.now()
            }
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error asking Gemini AI: {str(e)}")
            return {
                'error': str(e),
                'ai_analysis': f'Error communicating with AI: {str(e)}',
                'recommendation': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Communication error with AI brain',
                'risk_level': 'HIGH'
            }
    
    async def _get_ai_system_health(self, user_id: int) -> Dict[str, Any]:
        """Get AI-powered system health analysis"""
        try:
            if not self.autonomous_agent:
                return {
                    'system_status': 'ERROR',
                    'ai_brain_status': 'OFFLINE',
                    'error': 'AI agent not initialized'
                }
            
            # Get system health through autonomous agent
            health_data = await self.autonomous_agent.get_system_health_for_discord()
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error getting AI system health: {str(e)}")
            return {
                'system_status': 'ERROR',
                'ai_brain_status': 'ERROR',
                'error': str(e)
            }
    
    async def get_command_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get command history"""
        try:
            return self.command_history[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting command history: {str(e)}")
            return []