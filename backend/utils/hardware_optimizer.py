import os
import psutil
import gc
import logging
from typing import Dict, Any
import asyncio
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """
    Hardware optimization utilities for resource-constrained environments
    Optimized for ThinkPad T470: i5-7th gen, 8GB RAM, 256GB SSD
    """
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.config = self._load_hardware_config()
        self.memory_alerts_enabled = True
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
                'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def _load_hardware_config(self) -> Dict[str, Any]:
        """Load hardware optimization configuration"""
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "hardware_optimized_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading hardware config: {e}")
            return {}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'system_total_gb': round(memory.total / (1024**3), 2),
                'system_used_gb': round(memory.used / (1024**3), 2),
                'system_available_gb': round(memory.available / (1024**3), 2),
                'system_percent': memory.percent,
                'process_memory_mb': round(process.memory_info().rss / (1024**2), 2),
                'process_memory_percent': round(process.memory_percent(), 2)
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get current CPU usage statistics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return {}
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get current disk usage statistics"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'percent': round((disk.used / disk.total) * 100, 2)
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}
    
    def check_resource_limits(self) -> Dict[str, bool]:
        """Check if system resources are within safe limits"""
        try:
            memory = self.get_memory_usage()
            cpu = self.get_cpu_usage()
            disk = self.get_disk_usage()
            
            config_thresholds = self.config.get('performance_monitoring', {})
            
            return {
                'memory_ok': memory.get('system_percent', 0) < (config_thresholds.get('alert_memory_threshold', 0.85) * 100),
                'cpu_ok': cpu.get('cpu_percent', 0) < (config_thresholds.get('alert_cpu_threshold', 0.9) * 100),
                'disk_ok': disk.get('percent', 0) < (config_thresholds.get('alert_disk_threshold', 0.8) * 100),
                'process_memory_ok': memory.get('process_memory_mb', 0) < self.config.get('microstructure_optimization', {}).get('redis_max_memory', '512mb').replace('mb', ''),
            }
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return {}
    
    def optimize_python_memory(self):
        """Optimize Python memory usage"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Set memory growth limit for Redis if configured
            redis_max_mem = self.config.get('microstructure_optimization', {}).get('redis_max_memory', '512mb')
            logger.info(f"Memory optimization applied - Redis limit: {redis_max_mem}")
            
        except Exception as e:
            logger.error(f"Error optimizing Python memory: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data files to save disk space"""
        try:
            cleanup_config = self.config.get('data_cleanup', {})
            if not cleanup_config.get('auto_cleanup_enabled', False):
                return
            
            # Clean up log files
            log_retention_days = cleanup_config.get('cleanup_interval_hours', 6) / 24
            self._cleanup_logs(log_retention_days)
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    def _cleanup_logs(self, retention_days: float):
        """Clean up old log files"""
        try:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            if logs_dir.exists():
                current_time = asyncio.get_event_loop().time()
                retention_seconds = retention_days * 24 * 3600
                
                for log_file in logs_dir.glob("*.log"):
                    if (current_time - log_file.stat().st_mtime) > retention_seconds:
                        log_file.unlink()
                        logger.debug(f"Deleted old log file: {log_file}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_patterns = ["*.tmp", "*.cache", "*.pid"]
            project_root = Path(__file__).parent.parent.parent
            
            for pattern in temp_patterns:
                for temp_file in project_root.rglob(pattern):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Deleted temp file: {temp_file}")
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal number of worker processes for this hardware"""
        try:
            cpu_cores = self.system_info.get('cpu_count', 4)
            max_workers = self.config.get('api_optimization', {}).get('max_workers', 4)
            
            # For i5-7th gen, use conservative worker count to avoid overload
            optimal = min(cpu_cores, max_workers)
            logger.info(f"Optimal worker count: {optimal}")
            return optimal
            
        except Exception as e:
            logger.error(f"Error calculating optimal worker count: {e}")
            return 2
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for processing based on available memory"""
        try:
            available_memory_gb = self.system_info.get('memory_total_gb', 8)
            
            # Conservative batch sizing for 8GB RAM
            if available_memory_gb <= 8:
                return 32
            elif available_memory_gb <= 16:
                return 64
            else:
                return 128
                
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return 32
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Get comprehensive resource monitoring data"""
        try:
            return {
                'timestamp': asyncio.get_event_loop().time(),
                'memory': self.get_memory_usage(),
                'cpu': self.get_cpu_usage(),
                'disk': self.get_disk_usage(),
                'limits_check': self.check_resource_limits(),
                'system_info': self.system_info,
                'recommendations': self._get_performance_recommendations()
            }
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {}
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations based on current resource usage"""
        try:
            recommendations = []
            limits = self.check_resource_limits()
            
            if not limits.get('memory_ok', True):
                recommendations.append("High memory usage detected. Consider reducing tick buffer size or enabling memory cleanup.")
            
            if not limits.get('cpu_ok', True):
                recommendations.append("High CPU usage detected. Consider increasing tick aggregation window or reducing concurrent processes.")
            
            if not limits.get('disk_ok', True):
                recommendations.append("Low disk space detected. Enable auto-cleanup or manually clean old data files.")
            
            if not limits.get('process_memory_ok', True):
                recommendations.append("Process memory usage high. Consider reducing Redis cache size or tick buffer.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def start_monitoring_loop(self, interval_seconds: int = 300):
        """Start continuous resource monitoring loop"""
        try:
            logger.info(f"Starting resource monitoring loop (interval: {interval_seconds}s)")
            
            while True:
                monitoring_data = self.monitor_resources()
                
                # Log warnings if resources are constrained
                limits = monitoring_data.get('limits_check', {})
                if not all(limits.values()):
                    logger.warning(f"Resource constraints detected: {limits}")
                    recommendations = monitoring_data.get('recommendations', [])
                    for rec in recommendations:
                        logger.warning(f"Recommendation: {rec}")
                
                # Trigger cleanup if needed
                if not limits.get('disk_ok', True):
                    self.cleanup_old_data()
                
                if not limits.get('memory_ok', True):
                    self.optimize_python_memory()
                
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
