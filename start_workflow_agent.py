#!/usr/bin/env python3
"""
ARIA Elite Multi-AI Workflow Agent Launcher
Autonomous AI assistant for system management, code review, and trading operations
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

from backend.core.gemini_workflow_agent import MultiAIWorkflowAgent
from backend.core.multi_ai_engine import MultiBrowserAIEngine
from backend.utils.config_loader import ConfigLoader
from backend.utils.logger import setup_logger
from backend.utils.discord_notifier import discord_notifier

logger = setup_logger(__name__)

class ARIAWorkflowSystem:
    """
    ARIA Elite Multi-AI Workflow System
    Autonomous AI assistant for institutional trading operations
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.workflow_agent: Optional[MultiAIWorkflowAgent] = None
        self.multi_ai_engine: Optional[MultiBrowserAIEngine] = None
        
        # System state
        self.start_time = datetime.now()
        self.initialized_components = []
        self.failed_components = []
        
    async def initialize_system(self):
        """Initialize the Multi-AI Workflow system"""
        logger.info("🧠 ARIA ELITE MULTI-AI WORKFLOW AGENT - INITIALIZING")
        logger.info("=" * 60)
        
        try:
            # 1. Initialize Multi-AI Browser Engine
            logger.info("🤖 Initializing Multi-AI Browser Engine...")
            self.multi_ai_engine = MultiBrowserAIEngine()
            await self.multi_ai_engine.initialize()
            self.initialized_components.append("Multi-AI Browser Engine")
            logger.info("✅ Multi-AI Browser Engine operational")
            
            # 2. Initialize Workflow Agent
            logger.info("🧠 Initializing Multi-AI Workflow Agent...")
            self.workflow_agent = MultiAIWorkflowAgent()
            await self.workflow_agent.initialize()
            self.initialized_components.append("Workflow Agent")
            logger.info("✅ Workflow Agent ready for autonomous operations")
            
            # System initialization complete
            self.is_running = True
            logger.info("=" * 60)
            logger.info("🎯 ARIA ELITE WORKFLOW AGENT - FULLY OPERATIONAL")
            logger.info(f"✅ Components initialized: {', '.join(self.initialized_components)}")
            if self.failed_components:
                logger.warning(f"⚠️ Failed components: {', '.join(self.failed_components)}")
            logger.info("=" * 60)
            
            # Send Discord notification
            await discord_notifier.send_system_startup(
                components_initialized=self.initialized_components,
                failed_components=self.failed_components,
                start_time=self.start_time,
                system_name="Workflow Agent"
            )
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Workflow system initialization failed: {str(e)}")
            raise
    
    async def run_workflow_loop(self):
        """Main workflow loop - autonomous AI operations"""
        logger.info("🔄 Starting Multi-AI workflow loop...")
        
        try:
            while not self.shutdown_event.is_set():
                # Process system diagnostics
                await self._process_system_diagnostics()
                
                # Handle code review tasks
                await self._process_code_review_tasks()
                
                # Process AI workflow tasks
                await self._process_ai_tasks()
                
                # System optimization tasks
                await self._process_optimization_tasks()
                
                # Discord command processing
                await self._process_discord_commands()
                
                # Wait before next iteration
                await asyncio.sleep(10)  # 10-second loop
                
        except Exception as e:
            logger.error(f"❌ Error in workflow loop: {str(e)}")
            await self._emergency_shutdown()
    
    async def _process_system_diagnostics(self):
        """Process system diagnostic tasks"""
        try:
            if self.workflow_agent:
                # Check for system errors and generate fixes
                await self.workflow_agent._process_diagnostic_queue()
        except Exception as e:
            logger.error(f"❌ System diagnostics error: {str(e)}")
    
    async def _process_code_review_tasks(self):
        """Process code review and auto-fix tasks"""
        try:
            if self.workflow_agent:
                # Process code review queue
                await self.workflow_agent._process_code_review_queue()
        except Exception as e:
            logger.error(f"❌ Code review error: {str(e)}")
    
    async def _process_ai_tasks(self):
        """Process AI-generated tasks and signals"""
        try:
            if self.workflow_agent:
                # Process signal validation and analysis
                await self.workflow_agent._process_signal_queue()
        except Exception as e:
            logger.error(f"❌ AI task processing error: {str(e)}")
    
    async def _process_optimization_tasks(self):
        """Process system optimization tasks"""
        try:
            if self.workflow_agent:
                # Hardware optimization and system tuning
                await self.workflow_agent._optimize_system_performance()
        except Exception as e:
            logger.error(f"❌ Optimization error: {str(e)}")
    
    async def _process_discord_commands(self):
        """Process Discord remote control commands"""
        try:
            if self.workflow_agent:
                # Process system control queue
                await self.workflow_agent._process_system_control_queue()
        except Exception as e:
            logger.error(f"❌ Discord command processing error: {str(e)}")
    
    async def _emergency_shutdown(self):
        """Emergency workflow system shutdown"""
        logger.critical("🚨 WORKFLOW AGENT EMERGENCY SHUTDOWN")
        
        try:
            # Send critical alert
            await discord_notifier.send_critical_error(
                error="Workflow agent emergency shutdown initiated",
                component="WorkflowSystem"
            )
            
        except Exception as e:
            logger.error(f"❌ Error during emergency shutdown: {str(e)}")
        
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Graceful workflow system shutdown"""
        logger.info("🛑 Initiating workflow system shutdown...")
        
        try:
            # Stop workflow loop
            self.shutdown_event.set()
            
            # Stop all components
            if self.workflow_agent and hasattr(self.workflow_agent, 'shutdown'):
                await self.workflow_agent.shutdown()
            
            if self.multi_ai_engine and hasattr(self.multi_ai_engine, 'shutdown'):
                await self.multi_ai_engine.shutdown()
            
            logger.info("✅ Workflow system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")

# Global system instance
workflow_system: Optional[ARIAWorkflowSystem] = None

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    global workflow_system
    logger.info(f"📡 Received signal {signum} - initiating shutdown...")
    
    if workflow_system:
        asyncio.create_task(workflow_system.shutdown())

async def main():
    """Main entry point for ARIA Elite Workflow Agent"""
    global workflow_system
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        workflow_system = ARIAWorkflowSystem()
        await workflow_system.initialize_system()
        
        # Start workflow loop
        await workflow_system.run_workflow_loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"❌ CRITICAL WORKFLOW ERROR: {str(e)}")
        raise
    finally:
        if workflow_system:
            await workflow_system.shutdown()

if __name__ == "__main__":
    print("🧠 ARIA ELITE MULTI-AI WORKFLOW AGENT")
    print("=" * 50)
    print("Autonomous AI Assistant for Trading Operations")
    print("Initializing Multi-AI system...")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        sys.exit(1)