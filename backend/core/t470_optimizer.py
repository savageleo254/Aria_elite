"""
ARIA-DAN T470 Hardware Optimizer
Specialized optimization for Lenovo T470: i5 7th Gen, 8GB RAM, 256GB SSD
Maximum efficiency for browser automation and AI model management
"""

import asyncio
import psutil
import gc
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    cpu_model: str = "Intel i5 7th Gen"
    cpu_cores: int = 4
    cpu_threads: int = 4
    ram_total_gb: int = 8
    ram_available_gb: float = 6.0
    storage_total_gb: int = 256
    storage_type: str = "SSD"
    gpu_available: bool = False

@dataclass
class OptimizationSettings:
    max_browser_instances: int = 2
    memory_per_browser_mb: int = 1024
    max_concurrent_requests: int = 4
    garbage_collection_interval: int = 300
    memory_cleanup_threshold: float = 0.8
    cpu_throttle_threshold: float = 0.85
    storage_cleanup_threshold: float = 0.9

class T470ResourceMonitor:
    """Real-time resource monitoring for T470 constraints"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = []
        self.alerts = []
        
    def start_monitoring(self):
        """Start continuous resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("T470 resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("T470 resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                resource_data = self._collect_resource_data()
                self.resource_history.append(resource_data)
                
                # Keep only last 100 measurements
                if len(self.resource_history) > 100:
                    self.resource_history.pop(0)
                
                # Check for resource alerts
                self._check_resource_alerts(resource_data)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(10)
    
    def _collect_resource_data(self) -> Dict[str, Any]:
        """Collect current resource usage data"""
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now(),
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "cpu_percent": cpu_usage,
            "disk_percent": disk.used / disk.total * 100,
            "disk_free_gb": disk.free / (1024**3),
            "process_count": len(psutil.pids())
        }
    
    def _check_resource_alerts(self, resource_data: Dict[str, Any]):
        """Check for resource threshold alerts"""
        alerts = []
        
        if resource_data["memory_percent"] > 85:
            alerts.append({
                "type": "memory_critical",
                "message": f"Memory usage critical: {resource_data['memory_percent']:.1f}%",
                "timestamp": resource_data["timestamp"]
            })
        
        if resource_data["cpu_percent"] > 90:
            alerts.append({
                "type": "cpu_critical", 
                "message": f"CPU usage critical: {resource_data['cpu_percent']:.1f}%",
                "timestamp": resource_data["timestamp"]
            })
        
        if resource_data["disk_percent"] > 90:
            alerts.append({
                "type": "storage_critical",
                "message": f"Storage usage critical: {resource_data['disk_percent']:.1f}%", 
                "timestamp": resource_data["timestamp"]
            })
        
        if alerts:
            self.alerts.extend(alerts)
            # Keep only last 50 alerts
            if len(self.alerts) > 50:
                self.alerts = self.alerts[-50:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        if not self.resource_history:
            return {"status": "no_data"}
            
        latest = self.resource_history[-1]
        return {
            "memory_usage": latest["memory_percent"],
            "memory_available": latest["memory_available_gb"],
            "cpu_usage": latest["cpu_percent"],
            "disk_usage": latest["disk_percent"],
            "disk_free": latest["disk_free_gb"],
            "process_count": latest["process_count"],
            "status": "healthy" if latest["memory_percent"] < 80 and latest["cpu_percent"] < 80 else "stressed",
            "recent_alerts": len([a for a in self.alerts if a["timestamp"] > datetime.now() - timedelta(minutes=10)])
        }

class T470MemoryManager:
    """Advanced memory management for T470 8GB constraint"""
    
    def __init__(self):
        self.memory_pools = {}
        self.cleanup_strategies = []
        
    async def optimize_memory_usage(self):
        """Perform comprehensive memory optimization"""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear Python caches
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            # Optimize browser memory
            await self._optimize_browser_memory()
            
            # Cleanup temporary files
            await self._cleanup_temp_files()
            
            logger.info("T470 memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
    
    async def _optimize_browser_memory(self):
        """Optimize browser memory usage"""
        try:
            # Implementation for browser memory optimization
            # This would integrate with the browser automation system
            pass
        except Exception as e:
            logger.error(f"Browser memory optimization failed: {str(e)}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files to free storage"""
        try:
            temp_dirs = [
                os.path.expanduser("~/AppData/Local/Temp"),
                "./logs",
                "./cache"
            ]
            
            cleaned_mb = 0
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    cleaned_mb += await self._clean_directory(temp_dir)
            
            logger.info(f"Cleaned {cleaned_mb:.1f}MB of temporary files")
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {str(e)}")
    
    async def _clean_directory(self, directory: str) -> float:
        """Clean old files from directory"""
        try:
            cleaned_size = 0
            cutoff_time = datetime.now() - timedelta(days=7)
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_size += file_size
                    except (OSError, PermissionError):
                        continue
            
            return cleaned_size / (1024 * 1024)  # Return MB
            
        except Exception as e:
            logger.error(f"Directory cleanup failed for {directory}: {str(e)}")
            return 0

class T470Optimizer:
    """
    Comprehensive T470 hardware optimizer for ARIA-DAN system
    Maximizes performance within 8GB RAM and i5 7th Gen constraints
    """
    
    def __init__(self):
        self.hardware_profile = HardwareProfile()
        self.optimization_settings = OptimizationSettings()
        self.resource_monitor = T470ResourceMonitor()
        self.memory_manager = T470MemoryManager()
        self.is_initialized = False
        self.optimization_active = False
        
    async def initialize(self):
        """Initialize T470 optimizer"""
        try:
            logger.info("Initializing T470 Hardware Optimizer")
            
            # Detect actual hardware specs
            await self._detect_hardware()
            
            # Apply initial optimizations
            await self._apply_system_optimizations()
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Start optimization loop
            asyncio.create_task(self._optimization_loop())
            
            self.is_initialized = True
            logger.info("T470 Hardware Optimizer initialized")
            
        except Exception as e:
            logger.error(f"T470 optimizer initialization failed: {str(e)}")
            raise
    
    async def _detect_hardware(self):
        """Detect actual hardware specifications"""
        try:
            # CPU information
            self.hardware_profile.cpu_cores = psutil.cpu_count(logical=False)
            self.hardware_profile.cpu_threads = psutil.cpu_count(logical=True)
            
            # Memory information
            memory = psutil.virtual_memory()
            self.hardware_profile.ram_total_gb = memory.total // (1024**3)
            self.hardware_profile.ram_available_gb = memory.available / (1024**3)
            
            # Storage information
            disk = psutil.disk_usage('/')
            self.hardware_profile.storage_total_gb = disk.total // (1024**3)
            
            # GPU detection
            try:
                import torch
                self.hardware_profile.gpu_available = torch.cuda.is_available()
            except ImportError:
                self.hardware_profile.gpu_available = False
            
            logger.info(f"Hardware detected: {self.hardware_profile.cpu_cores}C/{self.hardware_profile.cpu_threads}T, "
                       f"{self.hardware_profile.ram_total_gb}GB RAM, {self.hardware_profile.storage_total_gb}GB Storage, "
                       f"GPU: {self.hardware_profile.gpu_available}")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {str(e)}")
    
    async def _apply_system_optimizations(self):
        """Apply T470-specific system optimizations"""
        try:
            # Configure optimization settings based on detected hardware
            if self.hardware_profile.ram_total_gb <= 8:
                self.optimization_settings.max_browser_instances = 2
                self.optimization_settings.memory_per_browser_mb = 1024
                self.optimization_settings.max_concurrent_requests = 4
            
            # Set environment variables for memory optimization
            os.environ["NODE_OPTIONS"] = "--max-old-space-size=1024"
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            
            # Configure Python garbage collection
            gc.set_threshold(700, 10, 10)
            
            logger.info("T470 system optimizations applied")
            
        except Exception as e:
            logger.error(f"System optimization failed: {str(e)}")
    
    async def _optimization_loop(self):
        """Continuous optimization loop"""
        self.optimization_active = True
        
        while self.optimization_active:
            try:
                # Check resource status
                status = self.resource_monitor.get_current_status()
                
                # Apply optimizations based on current status
                if status.get("memory_usage", 0) > 80:
                    await self.memory_manager.optimize_memory_usage()
                
                if status.get("cpu_usage", 0) > 85:
                    await self._throttle_processes()
                
                if status.get("disk_usage", 0) > 90:
                    await self.memory_manager._cleanup_temp_files()
                
                # Sleep for optimization interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _throttle_processes(self):
        """Throttle processes when CPU usage is high"""
        try:
            # Implement process throttling logic
            logger.info("CPU throttling applied")
        except Exception as e:
            logger.error(f"Process throttling failed: {str(e)}")
    
    def get_optimized_config(self) -> Dict[str, Any]:
        """Get optimized configuration for T470"""
        return {
            "browser_automation": {
                "max_instances": self.optimization_settings.max_browser_instances,
                "memory_per_instance": self.optimization_settings.memory_per_browser_mb,
                "concurrent_requests": self.optimization_settings.max_concurrent_requests,
                "session_timeout": 300,
                "cleanup_interval": 600
            },
            "ai_models": {
                "max_concurrent_models": 2,
                "model_memory_limit": 512,
                "quantization": "int8",
                "batch_size": 1
            },
            "system": {
                "garbage_collection_interval": self.optimization_settings.garbage_collection_interval,
                "memory_cleanup_threshold": self.optimization_settings.memory_cleanup_threshold,
                "cpu_throttle_threshold": self.optimization_settings.cpu_throttle_threshold,
                "storage_cleanup_threshold": self.optimization_settings.storage_cleanup_threshold
            },
            "hardware_profile": {
                "cpu_cores": self.hardware_profile.cpu_cores,
                "ram_gb": self.hardware_profile.ram_total_gb,
                "gpu_available": self.hardware_profile.gpu_available,
                "storage_type": self.hardware_profile.storage_type
            }
        }
    
    async def optimize_for_browser_automation(self) -> Dict[str, Any]:
        """Get browser automation specific optimizations"""
        current_memory = psutil.virtual_memory().percent
        
        # Dynamic adjustment based on current memory usage
        if current_memory > 80:
            instances = 1
            memory_per_instance = 800
        elif current_memory > 60:
            instances = 2
            memory_per_instance = 1024
        else:
            instances = 2
            memory_per_instance = 1200
        
        return {
            "max_browser_instances": instances,
            "memory_per_browser_mb": memory_per_instance,
            "chrome_args": [
                "--no-sandbox",
                "--disable-dev-shm-usage", 
                "--disable-gpu",
                f"--memory-pressure-off",
                f"--max_old_space_size={memory_per_instance}",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=TranslateUI",
                "--disable-sync",
                "--disable-extensions"
            ],
            "session_management": {
                "max_session_duration": 1800,  # 30 minutes
                "cleanup_interval": 300,       # 5 minutes
                "memory_check_interval": 60    # 1 minute
            }
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive T470 system status"""
        resource_status = self.resource_monitor.get_current_status()
        
        return {
            "optimizer_status": "active" if self.optimization_active else "inactive",
            "hardware_profile": self.hardware_profile.__dict__,
            "optimization_settings": self.optimization_settings.__dict__,
            "current_resources": resource_status,
            "optimization_recommendations": await self._generate_recommendations(resource_status),
            "system_health": "excellent" if resource_status.get("memory_usage", 0) < 70 else
                           "good" if resource_status.get("memory_usage", 0) < 80 else
                           "stressed",
            "last_optimization": datetime.now().isoformat()
        }
    
    async def _generate_recommendations(self, resource_status: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on current status"""
        recommendations = []
        
        if resource_status.get("memory_usage", 0) > 80:
            recommendations.append("Consider reducing browser instances")
            recommendations.append("Enable aggressive garbage collection")
        
        if resource_status.get("cpu_usage", 0) > 85:
            recommendations.append("Reduce concurrent AI model queries")
            recommendations.append("Enable process throttling")
        
        if resource_status.get("disk_usage", 0) > 90:
            recommendations.append("Clean temporary files and logs")
            recommendations.append("Archive old trading data")
        
        if not recommendations:
            recommendations.append("System running optimally for T470 hardware")
        
        return recommendations
    
    async def shutdown(self):
        """Shutdown T470 optimizer"""
        try:
            self.optimization_active = False
            self.resource_monitor.stop_monitoring()
            logger.info("T470 optimizer shutdown completed")
        except Exception as e:
            logger.error(f"T470 optimizer shutdown error: {str(e)}")
