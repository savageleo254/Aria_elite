#!/usr/bin/env python3
"""
ARIA ELITE - Discord Notification Utility
Sends alerts and patches to Discord channels via webhooks
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp

logger = logging.getLogger(__name__)

class DiscordNotifier:
    """Discord webhook notification system for Gemini audit alerts and patches"""
    
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.backup_webhook_url = os.getenv("DISCORD_BACKUP_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
        self.max_message_length = 2000
        self.session = None
        
        if not self.enabled:
            logger.warning("Discord notifications disabled: DISCORD_WEBHOOK_URL not configured")
    
    async def initialize(self):
        """Initialize HTTP session"""
        if self.enabled:
            self.session = aiohttp.ClientSession()
            logger.info("Discord notifier initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def send_audit_alert(self, severity: str, component: str, issue: str, details: Dict[str, Any] = None) -> bool:
        """Send audit alert to Discord"""
        if not self.enabled:
            return False
        
        color = self._get_color_for_severity(severity)
        
        embed = {
            "title": f"🚨 ARIA ELITE Audit Alert",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Severity", "value": severity.upper(), "inline": True},
                {"name": "Component", "value": component, "inline": True},
                {"name": "Issue", "value": issue[:1000], "inline": False}
            ]
        }
        
        if details:
            for key, value in details.items():
                if len(embed["fields"]) < 25:  # Discord limit
                    embed["fields"].append({
                        "name": key.replace("_", " ").title(),
                        "value": str(value)[:1000],
                        "inline": True
                    })
        
        return await self._send_webhook(embeds=[embed])
    
    async def send_patch_notification(self, action: str, files_modified: List[str], summary: str, 
                                    patch_details: Dict[str, Any] = None) -> bool:
        """Send patch notification to Discord"""
        if not self.enabled:
            return False
        
        files_list = "\n".join([f"• `{file}`" for file in files_modified[:10]])
        if len(files_modified) > 10:
            files_list += f"\n... and {len(files_modified) - 10} more files"
        
        embed = {
            "title": f"🔧 Gemini Auto-Patch Applied",
            "color": 0x00ff00,  # Green
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Action", "value": action, "inline": True},
                {"name": "Files Modified", "value": files_list or "None", "inline": False},
                {"name": "Summary", "value": summary[:1000], "inline": False}
            ]
        }
        
        if patch_details:
            if patch_details.get("tests_passed"):
                embed["fields"].append({
                    "name": "Tests", "value": f"✅ {patch_details['tests_passed']} passed", "inline": True
                })
            
            if patch_details.get("performance_impact"):
                embed["fields"].append({
                    "name": "Performance", "value": patch_details["performance_impact"], "inline": True
                })
        
        return await self._send_webhook(embeds=[embed])
    
    async def send_system_status(self, status: Dict[str, Any]) -> bool:
        """Send system status update to Discord"""
        if not self.enabled:
            return False
        
        color = 0x00ff00 if status.get("healthy", True) else 0xff0000
        
        embed = {
            "title": "📊 ARIA ELITE System Status",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": []
        }
        
        for component, health in status.items():
            if isinstance(health, dict):
                status_text = "✅ Healthy" if health.get("status") == "healthy" else "❌ Issues"
                embed["fields"].append({
                    "name": component.replace("_", " ").title(),
                    "value": status_text,
                    "inline": True
                })
        
        return await self._send_webhook(embeds=[embed])
    
    async def send_critical_error(self, error: str, component: str, traceback: str = None) -> bool:
        """Send critical error notification to Discord"""
        if not self.enabled:
            return False
        
        embed = {
            "title": "💥 CRITICAL ERROR - Immediate Attention Required",
            "color": 0xff0000,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Component", "value": component, "inline": True},
                {"name": "Error", "value": error[:1000], "inline": False}
            ]
        }
        
        if traceback:
            # Truncate traceback to fit Discord limits
            truncated_tb = traceback[:800] + "..." if len(traceback) > 800 else traceback
            embed["fields"].append({
                "name": "Traceback", 
                "value": f"```python\n{truncated_tb}\n```", 
                "inline": False
            })
        
        # Try both primary and backup webhook for critical errors
        success = await self._send_webhook(embeds=[embed])
        if not success and self.backup_webhook_url:
            success = await self._send_webhook(embeds=[embed], use_backup=True)
        
        return success
    
    async def send_trading_alert(self, alert_type: str, symbol: str, message: str, 
                               signal_data: Dict[str, Any] = None) -> bool:
        """Send trading-related alerts to Discord"""
        if not self.enabled:
            return False
        
        color_map = {
            "signal": 0x0099ff,     # Blue
            "execution": 0xffaa00,  # Orange
            "risk": 0xff0000,       # Red
            "profit": 0x00ff00      # Green
        }
        
        embed = {
            "title": f"📈 Trading Alert - {alert_type.upper()}",
            "color": color_map.get(alert_type, 0x808080),
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Alert", "value": message, "inline": False}
            ]
        }
        
        if signal_data:
            if signal_data.get("direction"):
                embed["fields"].append({
                    "name": "Direction", 
                    "value": signal_data["direction"].upper(), 
                    "inline": True
                })
            
            if signal_data.get("confidence"):
                embed["fields"].append({
                    "name": "Confidence", 
                    "value": f"{signal_data['confidence']:.1%}", 
                    "inline": True
                })
        
        return await self._send_webhook(embeds=[embed])
    
    def _get_color_for_severity(self, severity: str) -> int:
        """Get Discord embed color based on severity"""
        colors = {
            "critical": 0xff0000,  # Red
            "high": 0xff6600,      # Orange
            "medium": 0xffcc00,    # Yellow
            "low": 0x00ff00,       # Green
            "info": 0x0099ff       # Blue
        }
        return colors.get(severity.lower(), 0x808080)  # Default gray
    
    async def _send_webhook(self, embeds: List[Dict], use_backup: bool = False) -> bool:
        """Send webhook message to Discord"""
        if not self.session:
            await self.initialize()
        
        webhook_url = self.backup_webhook_url if use_backup else self.webhook_url
        if not webhook_url:
            return False
        
        payload = {
            "username": "ARIA ELITE",
            "avatar_url": "https://cdn.discordapp.com/attachments/placeholder/aria_logo.png",
            "embeds": embeds
        }
        
        try:
            async with self.session.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 204:
                    return True
                elif response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get("retry-after", 1))
                    logger.warning(f"Discord rate limited, retrying in {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._send_webhook(embeds, use_backup)
                else:
                    logger.error(f"Discord webhook failed: {response.status} - {await response.text()}")
                    return False
        
        except Exception as e:
            logger.error(f"Discord webhook error: {str(e)}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Discord webhook connection"""
        if not self.enabled:
            logger.info("Discord notifications disabled")
            return False
        
        test_embed = {
            "title": "🧪 ARIA ELITE Connection Test",
            "description": "Discord notification system is operational",
            "color": 0x00ff00,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await self._send_webhook(embeds=[test_embed])
        if success:
            logger.info("Discord connection test successful")
        else:
            logger.error("Discord connection test failed")
        
        return success

# Global notifier instance
discord_notifier = DiscordNotifier()
