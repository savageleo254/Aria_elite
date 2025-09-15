import asyncio
import logging
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Supervisor:
    """
    Monitors system health, provides API for Gemini, and manages system state
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.is_running = False
        self.is_paused = False
        self.kill_switch_activated = False
        self.system_health = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_status": "unknown",
            "process_count": 0
        }
        self.service_status = {
            "gemini_agent": "unknown",
            "signal_manager": "unknown",
            "execution_engine": "unknown",
            "smc_module": "unknown",
            "news_scraper": "unknown",
            "ai_models": "unknown"
        }
        self.alerts = []
        self.metrics_history = []
        self.app = None
        
    async def initialize(self):
        """Initialize the supervisor"""
        try:
            logger.info("Initializing Supervisor")
            
            # Load configuration
            await self._load_config()
            
            # Create FastAPI app
            await self._create_api_app()
            
            # Start monitoring loops
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._service_monitoring_loop())
            asyncio.create_task(self._metrics_collection_loop())
            asyncio.create_task(self._alert_management_loop())
            
            self.is_initialized = True
            logger.info("Supervisor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supervisor: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load supervisor configuration"""
        try:
            self.project_config = self.config.load_project_config()
            self.supervisor_config = self.project_config.get("supervisor_api", {})
            
            self.bind_address = self.supervisor_config.get("bind", "127.0.0.1:9100")
            self.kill_switch_config = self.supervisor_config.get("kill_switch", {})
            
            logger.info("Supervisor configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load supervisor configuration: {str(e)}")
            raise
    
    async def _create_api_app(self):
        """Create FastAPI application for supervisor API"""
        try:
            app = FastAPI(
                title="ARIA Supervisor API",
                description="System monitoring and management API",
                version="1.0.0"
            )
            
            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Security
            security = HTTPBearer()
            
            async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
                token = credentials.credentials
                expected_token = os.getenv("SUPERVISOR_API_TOKEN")
                if token != expected_token:
                    raise HTTPException(status_code=401, detail="Invalid authentication token")
                return token
            
            # API Endpoints
            @app.get("/")
            async def root():
                return {"message": "ARIA Supervisor API", "version": "1.0.0"}
            
            @app.get("/health")
            async def health_check():
                return {
                    "status": "healthy" if not self.kill_switch_activated else "kill_switch_activated",
                    "timestamp": datetime.now().isoformat()
                }
            
            @app.get("/status")
            async def get_status(token: str = Depends(verify_token)):
                return await self.get_system_status()
            
            @app.get("/metrics")
            async def get_metrics(token: str = Depends(verify_token)):
                return await self.get_system_metrics()
            
            @app.get("/alerts")
            async def get_alerts(token: str = Depends(verify_token), limit: int = 100):
                return await self.get_alerts(limit)
            
            @app.post("/pause")
            async def pause_system(token: str = Depends(verify_token)):
                await self.pause_system()
                return {"message": "System paused successfully"}
            
            @app.post("/resume")
            async def resume_system(token: str = Depends(verify_token)):
                await self.resume_system()
                return {"message": "System resumed successfully"}
            
            @app.post("/kill")
            async def activate_kill_switch(token: str = Depends(verify_token)):
                await self.activate_kill_switch()
                return {"message": "Kill switch activated successfully"}
            
            @app.post("/restart")
            async def restart_system(token: str = Depends(verify_token)):
                await self.restart_system()
                return {"message": "System restart initiated"}
            
            @app.post("/restart_model")
            async def restart_model(model_type: str, token: str = Depends(verify_token)):
                await self.restart_model(model_type)
                return {"message": f"Model {model_type} restart initiated"}
            
            @app.post("/train")
            async def train_model(model_type: str, token: str = Depends(verify_token)):
                await self.train_model(model_type)
                return {"message": f"Model {model_type} training initiated"}
            
            @app.get("/logs")
            async def get_logs(
                service: Optional[str] = None,
                level: Optional[str] = None,
                limit: int = 100,
                token: str = Depends(verify_token)
            ):
                return await self.get_logs(service, level, limit)
            
            self.app = app
            
        except Exception as e:
            logger.error(f"Failed to create API app: {str(e)}")
            raise
    
    async def start_server(self):
        """Start the supervisor API server"""
        try:
            if not self.app:
                raise ValueError("API app not initialized")
            
            host, port = self.bind_address.split(":")
            
            logger.info(f"Starting supervisor API server on {self.bind_address}")
            
            config = uvicorn.Config(
                self.app,
                host=host,
                port=int(port),
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start supervisor server: {str(e)}")
            raise
    
    async def _health_monitoring_loop(self):
        """Monitor system health"""
        while True:
            try:
                await self._update_system_health()
                await self._check_kill_switch_conditions()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _update_system_health(self):
        """Update system health metrics"""
        try:
            # CPU usage
            self.system_health["cpu_usage"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_health["memory_usage"] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_health["disk_usage"] = disk.percent
            
            # Network status (simplified)
            self.system_health["network_status"] = "connected"
            
            # Process count
            self.system_health["process_count"] = len(psutil.pids())
            
            logger.debug(f"System health updated: CPU={self.system_health['cpu_usage']}%, "
                        f"Memory={self.system_health['memory_usage']}%, "
                        f"Disk={self.system_health['disk_usage']}%")
            
        except Exception as e:
            logger.error(f"Error updating system health: {str(e)}")
    
    async def _check_kill_switch_conditions(self):
        """Check if kill switch conditions are met"""
        try:
            if self.kill_switch_activated:
                return
            
            kill_conditions = []
            
            # Check CPU overload
            cpu_threshold = self.kill_switch_config.get("cpu_overload", "90%")
            cpu_limit = float(cpu_threshold.replace("%", ""))
            if self.system_health["cpu_usage"] > cpu_limit:
                kill_conditions.append(f"CPU usage {self.system_health['cpu_usage']}% > {cpu_limit}%")
            
            # Check memory overload
            memory_threshold = self.kill_switch_config.get("memory_overload", "85%")
            memory_limit = float(memory_threshold.replace("%", ""))
            if self.system_health["memory_usage"] > memory_limit:
                kill_conditions.append(f"Memory usage {self.system_health['memory_usage']}% > {memory_limit}%")
            
            # Check disk usage
            if self.system_health["disk_usage"] > 90:
                kill_conditions.append(f"Disk usage {self.system_health['disk_usage']}% > 90%")
            
            # Activate kill switch if conditions are met
            if kill_conditions:
                logger.warning(f"Kill switch conditions met: {', '.join(kill_conditions)}")
                await self.activate_kill_switch()
                
        except Exception as e:
            logger.error(f"Error checking kill switch conditions: {str(e)}")
    
    async def _service_monitoring_loop(self):
        """Monitor service status"""
        while True:
            try:
                await self._update_service_status()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in service monitoring loop: {str(e)}")
                await asyncio.sleep(120)
    
    async def _update_service_status(self):
        """Update status of all services"""
        try:
            # This would normally check actual service status
            # For now, simulate status checks
            
            services = [
                "gemini_agent",
                "signal_manager", 
                "execution_engine",
                "smc_module",
                "news_scraper",
                "ai_models"
            ]
            
            for service in services:
                # Simulate status check with random failures
                if self.kill_switch_activated:
                    self.service_status[service] = "stopped"
                elif self.is_paused:
                    self.service_status[service] = "paused"
                else:
                    # Random status for simulation
                    import random
                    status_options = ["running", "running", "running", "warning", "error"]
                    self.service_status[service] = random.choice(status_options)
            
            logger.debug("Service status updated")
            
        except Exception as e:
            logger.error(f"Error updating service status: {str(e)}")
    
    async def _metrics_collection_loop(self):
        """Collect and store metrics"""
        while True:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": self.system_health.copy(),
                    "service_status": self.service_status.copy(),
                    "is_paused": self.is_paused,
                    "kill_switch_activated": self.kill_switch_activated
                }
                
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                await asyncio.sleep(300)  # Collect every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(600)
    
    async def _alert_management_loop(self):
        """Manage alerts"""
        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in alert management loop: {str(e)}")
                await asyncio.sleep(120)
    
    async def _process_alerts(self):
        """Process and manage alerts"""
        try:
            # Check for new alerts based on system status
            new_alerts = []
            
            # CPU alerts
            if self.system_health["cpu_usage"] > 80:
                new_alerts.append({
                    "type": "warning",
                    "message": f"High CPU usage: {self.system_health['cpu_usage']}%",
                    "timestamp": datetime.now().isoformat(),
                    "source": "system_health"
                })
            
            # Memory alerts
            if self.system_health["memory_usage"] > 80:
                new_alerts.append({
                    "type": "warning",
                    "message": f"High memory usage: {self.system_health['memory_usage']}%",
                    "timestamp": datetime.now().isoformat(),
                    "source": "system_health"
                })
            
            # Service alerts
            for service, status in self.service_status.items():
                if status == "error":
                    new_alerts.append({
                        "type": "error",
                        "message": f"Service {service} is in error state",
                        "timestamp": datetime.now().isoformat(),
                        "source": "service_monitor"
                    })
                elif status == "warning":
                    new_alerts.append({
                        "type": "warning",
                        "message": f"Service {service} is in warning state",
                        "timestamp": datetime.now().isoformat(),
                        "source": "service_monitor"
                    })
            
            # Add new alerts
            for alert in new_alerts:
                self.alerts.append(alert)
                logger.warning(f"Alert generated: {alert['message']}")
            
            # Keep only last 500 alerts
            if len(self.alerts) > 500:
                self.alerts = self.alerts[-500:]
            
        except Exception as e:
            logger.error(f"Error processing alerts: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "system_health": self.system_health,
                "service_status": self.service_status,
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "kill_switch_activated": self.kill_switch_activated,
                "alert_count": len(self.alerts),
                "metrics_count": len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {}
    
    async def get_system_metrics(self, limit: int = 100) -> Dict[str, Any]:
        """Get system metrics history"""
        try:
            return {
                "metrics": self.metrics_history[-limit:],
                "total_metrics": len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    async def get_alerts(self, limit: int = 100) -> Dict[str, Any]:
        """Get alerts"""
        try:
            return {
                "alerts": self.alerts[-limit:],
                "total_alerts": len(self.alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return {}
    
    async def pause_system(self):
        """Pause the system"""
        try:
            self.is_paused = True
            logger.info("System paused")
            
            # Add alert
            self.alerts.append({
                "type": "info",
                "message": "System paused by user",
                "timestamp": datetime.now().isoformat(),
                "source": "user_action"
            })
            
        except Exception as e:
            logger.error(f"Error pausing system: {str(e)}")
            raise
    
    async def resume_system(self):
        """Resume the system"""
        try:
            self.is_paused = False
            logger.info("System resumed")
            
            # Add alert
            self.alerts.append({
                "type": "info",
                "message": "System resumed by user",
                "timestamp": datetime.now().isoformat(),
                "source": "user_action"
            })
            
        except Exception as e:
            logger.error(f"Error resuming system: {str(e)}")
            raise
    
    async def activate_kill_switch(self):
        """Activate kill switch"""
        try:
            self.kill_switch_activated = True
            self.is_paused = True
            
            logger.warning("Kill switch activated!")
            
            # Add critical alert
            self.alerts.append({
                "type": "critical",
                "message": "Kill switch activated - System stopped",
                "timestamp": datetime.now().isoformat(),
                "source": "kill_switch"
            })
            
            # Here you would normally close all positions and stop all trading
            # This is a safety mechanism
            
        except Exception as e:
            logger.error(f"Error activating kill switch: {str(e)}")
            raise
    
    async def restart_system(self):
        """Restart the system"""
        try:
            logger.info("System restart initiated")
            
            # Add alert
            self.alerts.append({
                "type": "info",
                "message": "System restart initiated",
                "timestamp": datetime.now().isoformat(),
                "source": "user_action"
            })
            
            # This would normally restart the entire system
            # For now, just reset some states
            self.is_paused = False
            self.kill_switch_activated = False
            
        except Exception as e:
            logger.error(f"Error restarting system: {str(e)}")
            raise
    
    async def restart_model(self, model_type: str):
        """Restart a specific model"""
        try:
            logger.info(f"Model {model_type} restart initiated")
            
            # Add alert
            self.alerts.append({
                "type": "info",
                "message": f"Model {model_type} restart initiated",
                "timestamp": datetime.now().isoformat(),
                "source": "user_action"
            })
            
            # This would normally restart the specific model service
            
        except Exception as e:
            logger.error(f"Error restarting model {model_type}: {str(e)}")
            raise
    
    async def train_model(self, model_type: str):
        """Train a specific model"""
        try:
            logger.info(f"Model {model_type} training initiated")
            
            # Add alert
            self.alerts.append({
                "type": "info",
                "message": f"Model {model_type} training initiated",
                "timestamp": datetime.now().isoformat(),
                "source": "user_action"
            })
            
            # This would normally trigger model training
            
        except Exception as e:
            logger.error(f"Error training model {model_type}: {str(e)}")
            raise
    
    async def get_logs(self, service: Optional[str] = None, level: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Get system logs"""
        try:
            # This would normally read from log files
            # For now, return mock logs
            
            mock_logs = []
            for i in range(min(limit, 50)):
                log_entry = {
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "level": random.choice(["INFO", "WARNING", "ERROR"]),
                    "service": service or random.choice(["gemini_agent", "signal_manager", "execution_engine"]),
                    "message": f"Mock log entry {i}"
                }
                
                if level and log_entry["level"] != level:
                    continue
                
                mock_logs.append(log_entry)
            
            return {
                "logs": mock_logs,
                "total_logs": len(mock_logs)
            }
            
        except Exception as e:
            logger.error(f"Error getting logs: {str(e)}")
            return {}

# Import random for mock logs
import random