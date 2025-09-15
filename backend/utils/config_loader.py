import json
import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Utility class for loading configuration files
    """
    
    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
    
    def get_database_url(self) -> str:
        """Get database URL from configuration"""
        try:
            project_config = self.load_project_config()
            return project_config.get("database", {}).get("url", "sqlite:///./db/custom.db")
        except Exception as e:
            logger.error(f"Error getting database URL: {str(e)}")
            return "sqlite:///./db/custom.db"
        
    def load_project_config(self) -> Dict[str, Any]:
        """Load project configuration"""
        try:
            config_path = os.path.join(self.config_dir, 'project_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading project config: {str(e)}")
            raise
    
    def load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy configuration"""
        try:
            config_path = os.path.join(self.config_dir, 'strategy_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading strategy config: {str(e)}")
            raise
    
    def load_execution_config(self) -> Dict[str, Any]:
        """Load execution configuration"""
        try:
            config_path = os.path.join(self.config_dir, 'execution_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading execution config: {str(e)}")
            raise
    
    def load_ai_accounts_config(self) -> Dict[str, Any]:
        """Load AI accounts configuration"""
        try:
            config_path = os.path.join(self.config_dir, 'ai_accounts_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading AI accounts config: {str(e)}")
            # Return default configuration if file doesn't exist
            return {
                "ai_accounts": {},
                "rotation_strategy": "round_robin",
                "max_concurrent_sessions": 5,
                "session_timeout": 1800,
                "provider_preferences": {
                    "primary": "chatgpt",
                    "secondary": ["gemini", "claude", "grok", "perplexity"],
                    "fallback_order": ["chatgpt", "gemini", "claude", "grok", "perplexity"]
                },
                "performance_optimization": {
                    "enable_caching": True,
                    "cache_ttl": 300,
                    "parallel_requests": True,
                    "max_retries": 3,
                    "request_timeout": 60
                },
                "cost_optimization": {
                    "prefer_free_tiers": True,
                    "usage_thresholds": {
                        "chatgpt": 0.8,
                        "gemini": 0.9,
                        "claude": 0.85,
                        "grok": 0.95,
                        "perplexity": 0.9
                    },
                    "budget_alerts": True,
                    "daily_budget_limit": 100
                }
            }
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load any configuration file"""
        try:
            config_path = os.path.join(self.config_dir, filename)
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config {filename}: {str(e)}")
            raise
    
    def save_config(self, filename: str, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            config_path = os.path.join(self.config_dir, filename)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving config {filename}: {str(e)}")
            raise
