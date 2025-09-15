"""
ARIA-DAN Premium Browser Automation Engine
Pure browser-based access to GPT-5, Grok-4, Gemini-2.5-Pro, Claude-4, DeepSeek-R1
NO API KEYS - Full browser automation with intelligent rotation
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
import undetected_chromedriver as uc
import time
import hashlib
import os
from pathlib import Path
import re
import pickle
from selenium.webdriver.common.keys import Keys
import functools

logger = logging.getLogger(__name__)

class PremiumModelTier(Enum):
    # OpenAI (via ChatGPT web)
    GPT_5 = "gpt-5"
    GPT_4_TURBO_2024 = "gpt-4-turbo-2024"
    GPT_4O_LATEST = "gpt-4o-latest"
    
    # Google (via Gemini web)
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH_THINKING = "gemini-2.5-flash-thinking"
    GEMINI_EXP_1206 = "gemini-exp-1206"
    
    # xAI (via Grok web)
    GROK_4 = "grok-4"
    GROK_4_MINI = "grok-4-mini"
    
    # Anthropic (via Claude web)
    CLAUDE_4 = "claude-4"
    CLAUDE_3_7_SONNET_2024 = "claude-3.7-sonnet-2024"
    
    # DeepSeek (via web)
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_V3 = "deepseek-v3"
    
    # Perplexity (via web)
    PERPLEXITY_SONAR_PRO = "sonar-pro"
    PERPLEXITY_SONAR_HUGE = "sonar-huge"

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"

class AccountStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class SafeChrome(uc.Chrome):
    def __del__(self):
        try:
            self.quit()
        except:
            pass

class BrowserSession:
    def __init__(self, provider: str, account_email: str, account_password: str):
        self.provider = provider
        self.account_email = account_email
        self.account_password = account_password
        self.driver = None
        self.session_id = None
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.usage_count = 0
        self.status = AccountStatus.ACTIVE
        self.response_times = []
        self.success_rate = 1.0
        
    async def initialize(self):
        """Initialize browser session with fixed cookie handling"""
        try:
            # T470 optimized Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--memory-pressure-off')
            chrome_options.add_argument('--max_old_space_size=1024')
            chrome_options.add_argument('--disable-background-timer-throttling')
            chrome_options.add_argument('--disable-backgrounding-occluded-windows')
            chrome_options.add_argument('--disable-renderer-backgrounding')
            chrome_options.add_argument('--disable-features=TranslateUI')
            chrome_options.add_argument('--disable-ipc-flooding-protection')
            chrome_options.add_argument('--disable-background-networking')
            chrome_options.add_argument('--disable-sync')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Use undetected-chromedriver
            self.driver = SafeChrome(options=chrome_options, version_main=None)
            self.session_id = hashlib.md5(f"{self.provider}_{self.account_email}_{datetime.now()}".encode()).hexdigest()[:8]
            
            # Email-based cookie filename
            email_hash = hashlib.md5(self.account_email.encode()).hexdigest()[:8]
            cookie_file = f"{self.provider}_cookies_{email_hash}.pkl"
            
            # Navigate to provider domain before adding cookies
            domain_map = {
                "chatgpt": "https://chat.openai.com",
                "gemini": "https://gemini.google.com",
                "claude": "https://claude.ai",
                "grok": "https://x.ai"
            }
            start_url = domain_map.get(self.provider, "about:blank")
            try:
                self.driver.get(start_url)
            except Exception:
                pass  # Continue even if navigation fails
            
            if os.path.exists(cookie_file):
                with open(cookie_file, "rb") as file:
                    cookies = pickle.load(file)
                for cookie in cookies:
                    try:
                        cookie.pop('sameSite', None)
                        self.driver.add_cookie(cookie)
                    except Exception as e:
                        logger.debug(f"Failed to add cookie: {e}")
                logger.info(f"Loaded cookies for {self.provider} ({self.account_email})")
            
            logger.info(f"Browser session initialized: {self.session_id} for {self.provider}")
            return True
            
        except Exception as e:
            logger.exception(f"Browser initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up browser session"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
            logger.info(f"Browser session cleaned up: {self.session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up browser session: {str(e)}")

class IntelligentRotationManager:
    def __init__(self):
        self.provider_limits = {
            # OpenAI limits (estimated)
            "chatgpt": {
                PremiumModelTier.GPT_5: {"daily": 2, "hourly": 1, "session": "premium"},
                PremiumModelTier.GPT_4_TURBO_2024: {"daily": 10, "hourly": 3, "session": "standard"},
                PremiumModelTier.GPT_4O_LATEST: {"daily": 15, "hourly": 5, "session": "free"}
            },
            # Google limits
            "gemini": {
                PremiumModelTier.GEMINI_2_5_PRO: {"daily": 5, "hourly": 2, "session": "premium"},
                PremiumModelTier.GEMINI_2_5_FLASH_THINKING: {"daily": 5, "hourly": 2, "session": "thinking"},
                PremiumModelTier.GEMINI_EXP_1206: {"daily": 20, "hourly": 5, "session": "experimental"}
            },
            # xAI limits
            "grok": {
                PremiumModelTier.GROK_4: {"daily": 10, "hourly": 3, "session": "premium"},
                PremiumModelTier.GROK_4_MINI: {"daily": 25, "hourly": 8, "session": "standard"}
            },
            # Anthropic limits
            "claude": {
                PremiumModelTier.CLAUDE_4: {"daily": 8, "hourly": 2, "session": "premium"},
                PremiumModelTier.CLAUDE_3_7_SONNET_2024: {"daily": 15, "hourly": 4, "session": "standard"}
            },
            # DeepSeek limits
            "deepseek": {
                PremiumModelTier.DEEPSEEK_R1: {"daily": 20, "hourly": 5, "session": "reasoning"},
                PremiumModelTier.DEEPSEEK_V3: {"daily": 30, "hourly": 8, "session": "standard"}
            },
            # Perplexity limits
            "perplexity": {
                PremiumModelTier.PERPLEXITY_SONAR_PRO: {"daily": 100, "hourly": 20, "session": "pro"},
                PremiumModelTier.PERPLEXITY_SONAR_HUGE: {"daily": 50, "hourly": 10, "session": "huge"}
            }
        }
        
        self.account_usage = {}
        self.rotation_history = []
        
    def get_optimal_account(self, model: PremiumModelTier, complexity: TaskComplexity) -> Optional[Dict[str, Any]]:
        """Get optimal account for model and task complexity"""
        try:
            provider = self._get_provider_for_model(model)
            available_accounts = self._get_available_accounts(provider, model)
            
            if not available_accounts:
                return None
                
            # Priority scoring based on usage, success rate, and response time
            scored_accounts = []
            for account in available_accounts:
                score = self._calculate_account_score(account, model, complexity)
                scored_accounts.append((account, score))
            
            # Sort by score (higher is better)
            scored_accounts.sort(key=lambda x: x[1], reverse=True)
            
            return scored_accounts[0][0] if scored_accounts else None
            
        except Exception as e:
            logger.error(f"Error getting optimal account: {str(e)}")
            return None
    
    def _get_provider_for_model(self, model: PremiumModelTier) -> str:
        """Get provider name for model"""
        model_providers = {
            PremiumModelTier.GPT_5: "chatgpt",
            PremiumModelTier.GPT_4_TURBO_2024: "chatgpt",
            PremiumModelTier.GPT_4O_LATEST: "chatgpt",
            PremiumModelTier.GEMINI_2_5_PRO: "gemini",
            PremiumModelTier.GEMINI_2_5_FLASH_THINKING: "gemini",
            PremiumModelTier.GEMINI_EXP_1206: "gemini",
            PremiumModelTier.GROK_4: "grok",
            PremiumModelTier.GROK_4_MINI: "grok",
            PremiumModelTier.CLAUDE_4: "claude",
            PremiumModelTier.CLAUDE_3_7_SONNET_2024: "claude",
            PremiumModelTier.DEEPSEEK_R1: "deepseek",
            PremiumModelTier.DEEPSEEK_V3: "deepseek",
            PremiumModelTier.PERPLEXITY_SONAR_PRO: "perplexity",
            PremiumModelTier.PERPLEXITY_SONAR_HUGE: "perplexity"
        }
        return model_providers.get(model, "unknown")
    
    def _get_available_accounts(self, provider: str, model: PremiumModelTier) -> List[Dict[str, Any]]:
        """Get available accounts for provider and model"""
        # This would be loaded from secure config/database
        # For now, return placeholder structure
        return [
            {
                "email": f"account1@{provider}.com",
                "password": "secure_password_1",
                "provider": provider,
                "model": model,
                "daily_usage": 0,
                "hourly_usage": 0,
                "last_used": datetime.now() - timedelta(hours=1),
                "success_rate": 0.95,
                "avg_response_time": 2.5
            }
        ]
    
    def _calculate_account_score(self, account: Dict[str, Any], model: PremiumModelTier, complexity: TaskComplexity) -> float:
        """Calculate account priority score"""
        try:
            base_score = 100.0
            
            # Usage penalty
            limits = self.provider_limits.get(account["provider"], {}).get(model, {})
            daily_limit = limits.get("daily", 100)
            hourly_limit = limits.get("hourly", 10)
            
            usage_penalty = (account["daily_usage"] / daily_limit) * 30
            usage_penalty += (account["hourly_usage"] / hourly_limit) * 20
            
            # Success rate bonus
            success_bonus = account["success_rate"] * 20
            
            # Response time penalty (lower is better)
            response_penalty = min(account["avg_response_time"], 10) * 2
            
            # Complexity matching
            complexity_bonus = 0
            if complexity == TaskComplexity.CRITICAL and "premium" in limits.get("session", ""):
                complexity_bonus = 15
            elif complexity == TaskComplexity.COMPLEX and "pro" in limits.get("session", ""):
                complexity_bonus = 10
            
            final_score = base_score - usage_penalty + success_bonus - response_penalty + complexity_bonus
            return max(final_score, 0)
            
        except Exception as e:
            logger.error(f"Error calculating account score: {str(e)}")
            return 0.0

class PremiumBrowserEngine:
    """
    Pure browser automation engine for premium AI models
    NO API KEYS REQUIRED - Direct web interface automation
    """
    
    def __init__(self):
        self.rotation_manager = IntelligentRotationManager()
        self.active_sessions = {}
        self.is_initialized = False
        self.model_hierarchy = {
            "critical": [PremiumModelTier.GPT_5, PremiumModelTier.GEMINI_2_5_PRO, PremiumModelTier.CLAUDE_4],
            "complex": [PremiumModelTier.GROK_4, PremiumModelTier.GPT_4_TURBO_2024, PremiumModelTier.DEEPSEEK_R1],
            "standard": [PremiumModelTier.GPT_4O_LATEST, PremiumModelTier.GEMINI_2_5_FLASH_THINKING, PremiumModelTier.PERPLEXITY_SONAR_PRO],
            "efficient": [PremiumModelTier.GROK_4_MINI, PremiumModelTier.CLAUDE_3_7_SONNET_2024, PremiumModelTier.PERPLEXITY_SONAR_HUGE]
        }
        
        # T470 Hardware constraints
        self.max_concurrent_sessions = 2  # Limit for 8GB RAM
        self.session_memory_limit = 1024  # 1GB per session
        
    async def initialize(self):
        """Initialize the premium browser engine"""
        try:
            logger.info("Initializing Premium Browser Engine - NO API KEYS REQUIRED")
            
            # Load account configurations
            await self._load_account_configs()
            
            # Initialize base sessions for critical models
            await self._initialize_base_sessions()
            
            self.is_initialized = True
            logger.info("Premium Browser Engine initialized - Ready for institutional domination")
            
        except Exception as e:
            logger.error(f"Failed to initialize Premium Browser Engine: {str(e)}")
            raise
    
    async def _load_account_configs(self):
        """Load account configurations from secure storage"""
        try:
            # In production, load from encrypted config/database
            # For now, use environment variables as example
            self.account_configs = {
                "chatgpt": [
                    {
                        "email": os.getenv("CHATGPT_EMAIL_1", ""),
                        "password": os.getenv("CHATGPT_PASSWORD_1", ""),
                        "subscription": "premium"
                    },
                    {
                        "email": os.getenv("CHATGPT_EMAIL_2", ""),
                        "password": os.getenv("CHATGPT_PASSWORD_2", ""),
                        "subscription": "premium"
                    }
                ],
                "gemini": [
                    {
                        "email": os.getenv("GEMINI_EMAIL_1", ""),
                        "password": os.getenv("GEMINI_PASSWORD_1", ""),
                        "subscription": "premium"
                    }
                ],
                "grok": [
                    {
                        "email": os.getenv("GROK_EMAIL_1", ""),
                        "password": os.getenv("GROK_PASSWORD_1", ""),
                        "subscription": "premium"
                    }
                ],
                "claude": [
                    {
                        "email": os.getenv("CLAUDE_EMAIL_1", ""),
                        "password": os.getenv("CLAUDE_PASSWORD_1", ""),
                        "subscription": "premium"
                    }
                ]
            }
            
            logger.info(f"Account configurations loaded for {len(self.account_configs)} providers")
            
        except Exception as e:
            logger.error(f"Failed to load account configs: {str(e)}")
    
    async def _initialize_base_sessions(self):
        """Initialize base browser sessions for critical models"""
        try:
            # Initialize one session per critical provider (respecting T470 limits)
            critical_providers = ["chatgpt", "gemini", "claude"]  # Limit to 3 for T470
            
            for provider in critical_providers:
                if provider in self.account_configs and self.account_configs[provider]:
                    account = self.account_configs[provider][0]
                    session = BrowserSession(provider, account["email"], account["password"])
                    
                    if await session.initialize():
                        self.active_sessions[provider] = session
                        logger.info(f"Base session initialized for {provider}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize base sessions: {str(e)}")
    
    async def generate_signal(self, symbol: str, timeframe: str, market_context: Dict[str, Any], complexity: TaskComplexity) -> Dict[str, Any]:
        """Generate trading signal using premium models"""
        try:
            # Select optimal model based on complexity
            selected_model = self._select_optimal_model(complexity)
            
            # Get optimal account for the model
            account = self.rotation_manager.get_optimal_account(selected_model, complexity)
            if not account:
                raise Exception(f"No available account for {selected_model}")
            
            # Get or create browser session
            session = await self._get_or_create_session(selected_model, account)
            
            # Generate trading prompt
            trading_prompt = self._create_trading_prompt(symbol, timeframe, market_context, complexity)
            
            # Execute browser automation to get response
            response = await self._execute_browser_query(session, selected_model, trading_prompt)
            
            # Parse and validate response
            signal = await self._parse_signal_response(response, symbol, selected_model)
            
            model_str = selected_model.value if hasattr(selected_model, 'value') else str(selected_model)
            logger.info(f"Premium signal generated using {model_str} for {symbol}")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate premium signal: {str(e)}")
            # Fallback to basic signal
            return self._generate_fallback_signal(symbol, market_context)
    
    def _select_optimal_model(self, complexity: TaskComplexity) -> PremiumModelTier:
        """Select optimal model based on task complexity"""
        complexity_mapping = {
            TaskComplexity.CRITICAL: "critical",
            TaskComplexity.COMPLEX: "complex",
            TaskComplexity.MODERATE: "standard",
            TaskComplexity.SIMPLE: "efficient"
        }
        
        tier = complexity_mapping.get(complexity, "standard")
        available_models = self.model_hierarchy[tier]
        
        for model in available_models:
            if self._is_model_available(model):
                return model
        
        for tier_models in self.model_hierarchy.values():
            for model in tier_models:
                if self._is_model_available(model):
                    return model
        
        # Ultimate fallback - return enum member
        return PremiumModelTier.GPT_4O_LATEST
    
    def _is_model_available(self, model: PremiumModelTier) -> bool:
        """Check if model is available for use"""
        provider = self.rotation_manager._get_provider_for_model(model)
        return provider in self.account_configs and len(self.account_configs[provider]) > 0
    
    async def _get_or_create_session(self, model: PremiumModelTier, account: Dict[str, Any]) -> BrowserSession:
        """Get existing session or create new one"""
        provider = account["provider"]
        
        # Check if we have active session for this provider
        if provider in self.active_sessions:
            session = self.active_sessions[provider]
            if session.status == AccountStatus.ACTIVE:
                return session
        
        # Create new session (respecting T470 limits)
        if len(self.active_sessions) >= self.max_concurrent_sessions:
            # Clean up oldest session
            await self._cleanup_oldest_session()
        
        session = BrowserSession(provider, account["email"], account["password"])
        if await session.initialize():
            self.active_sessions[provider] = session
            return session
        else:
            raise Exception(f"Failed to create session for {provider}")
    
    def _create_trading_prompt(self, symbol: str, timeframe: str, market_context: Dict[str, Any], complexity: TaskComplexity) -> str:
        """Create specialized trading prompt for premium models"""
        return f"""
        INSTITUTIONAL TRADING SIGNAL ANALYSIS - ARIA-DAN PROTOCOL
        
        Symbol: {symbol}
        Timeframe: {timeframe}
        Analysis Complexity: {complexity.value.upper()}
        
        Market Context:
        - Current Price: {market_context.get('current_price', 'N/A')}
        - Trend: {market_context.get('trend', 'sideways')}
        - Volatility: {market_context.get('volatility', 0.001):.4f}
        - RSI: {market_context.get('rsi', 50.0):.2f}
        - Volume: {market_context.get('volume', 0)}
        
        Required Analysis:
        1. Market structure analysis (support/resistance, trends)
        2. Multi-timeframe confluence
        3. Risk-reward assessment
        4. Entry/exit strategy
        5. Position sizing recommendation
        
        Respond ONLY in JSON format:
        {{
            "signal": "BUY|SELL|HOLD",
            "confidence": 0.0-1.0,
            "entry_price": float,
            "stop_loss": float,
            "take_profit": float,
            "position_size": 0.01-0.05,
            "reasoning": "detailed analysis",
            "risk_level": "LOW|MEDIUM|HIGH",
            "timeframe_bias": "BULLISH|BEARISH|NEUTRAL"
        }}
        """
    
    async def _execute_browser_query(self, session: BrowserSession, model: PremiumModelTier, prompt: str) -> str:
        """Execute browser automation to query the model"""
        try:
            provider = self.rotation_manager._get_provider_for_model(model)
            
            # Provider-specific automation
            if provider == "chatgpt":
                return await self._execute_chatgpt_query(session, prompt)
            elif provider == "gemini":
                return await self._execute_gemini_query(session, prompt)
            elif provider == "grok":
                return await self._execute_grok_query(session, prompt)
            elif provider == "claude":
                return await self._execute_claude_query(session, prompt)
            elif provider == "deepseek":
                return await self._execute_deepseek_query(session, prompt)
            elif provider == "perplexity":
                return await self._execute_perplexity_query(session, prompt)
            else:
                raise Exception(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Browser query execution failed: {str(e)}")
            session.status = AccountStatus.ERROR
            return ""
    
    async def _execute_chatgpt_query(self, session: BrowserSession, prompt: str) -> str:
        """Execute ChatGPT web interface automation"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 30)
            
            # Navigate to ChatGPT
            driver.get("https://chat.openai.com")
            
            # Wait for and find textarea
            textarea = wait.until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))
            
            # Clear and send prompt
            textarea.clear()
            textarea.send_keys(prompt)
            
            # Find and click send button
            send_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='send-button']")))
            send_button.click()
            
            # Wait for response
            await asyncio.sleep(5)  # Initial wait
            response_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-message-author-role='assistant']")))
            
            # Extract response text
            response_text = response_div.text
            
            # Update session stats
            session.usage_count += 1
            session.last_used = datetime.now()
            
            return response_text
            
        except Exception as e:
            logger.error(f"ChatGPT query failed: {str(e)}")
            return ""
    
    async def _execute_gemini_query(self, session: BrowserSession, prompt: str) -> str:
        """Execute Gemini web interface automation with streaming response"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 30)
            
            # Check if already on Gemini app
            if "gemini.google.com/app" not in driver.current_url:
                driver.get("https://gemini.google.com/app")
                
            # Wait for prompt input
            textarea = wait.until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Enter a prompt here']")))
            textarea.clear()
            
            # Type prompt character-by-character (human-like)
            for char in prompt:
                textarea.send_keys(char)
                await asyncio.sleep(0.05)
                
            # Submit prompt
            textarea.send_keys(Keys.ENTER)
            
            # Capture streaming response
            full_response = ""
            last_response_length = 0
            retry_count = 0
            
            while retry_count < 5:
                try:
                    response_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-message-text]")))
                    current_response = response_div.text
                    
                    # Check if response is growing
                    if len(current_response) > last_response_length:
                        full_response = current_response
                        last_response_length = len(full_response)
                        retry_count = 0
                    else:
                        retry_count += 1
                        
                    # Check for completion markers
                    if "●" not in full_response and "..." not in full_response:
                        break
                        
                    await asyncio.sleep(1)
                except:
                    retry_count += 1
                    await asyncio.sleep(1)
            
            # Update session stats
            session.usage_count += 1
            session.last_used = datetime.now()
            
            return full_response
            
        except Exception as e:
            logger.error(f"Gemini query failed: {str(e)}")
            return ""
    
    async def _gemini_login(self, session: BrowserSession):
        """Full Gemini login flow with credential handling"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 20)
            
            # Navigate to Gemini login
            driver.get("https://accounts.google.com/ServiceLogin?service=cloudconsole&passive=1209600&continue=https://gemini.google.com/app&followup=https://gemini.google.com/app")
            
            # Enter email
            email_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']")))
            email_field.clear()
            email_field.send_keys(session.account_email)
            email_field.send_keys(Keys.RETURN)
            
            # Enter password
            password_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
            password_field.clear()
            password_field.send_keys(session.account_password)
            password_field.send_keys(Keys.RETURN)
            
            # Handle 2FA if present
            try:
                twofa_prompt = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(text(),'2-Step Verification')]"))
                )
                logger.warning("Gemini 2FA detected - manual intervention required")
                # Save screenshot for user
                driver.save_screenshot(f"gemini_2fa_{session.session_id}.png")
                return False
            except TimeoutException:
                pass  # No 2FA
            
            # Wait for Gemini app to load
            wait.until(EC.url_contains("gemini.google.com/app"))
            logger.info(f"Gemini login successful for {session.account_email}")
            
            # Save cookies with email hash
            email_hash = hashlib.md5(session.account_email.encode()).hexdigest()[:8]
            cookie_file = f"gemini_cookies_{email_hash}.pkl"
            with open(cookie_file, "wb") as file:
                pickle.dump(driver.get_cookies(), file)
                
            return True
            
        except Exception as e:
            logger.error(f"Gemini login failed: {str(e)}")
            return False
    
    async def _claude_login(self, session: BrowserSession):
        """Full Claude login flow with credential handling"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 20)
            
            # Navigate to Claude login
            driver.get("https://claude.ai/login")
            
            # Enter email
            email_field = wait.until(EC.presence_of_element_located((By.ID, "email")))
            email_field.clear()
            email_field.send_keys(session.account_email)
            
            # Enter password
            password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
            password_field.clear()
            password_field.send_keys(session.account_password)
            
            # Submit
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Login')]")))
            login_button.click()
            
            # Handle 2FA if present
            try:
                twofa_field = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.NAME, "otp"))
                )
                logger.warning("Claude 2FA detected - manual intervention required")
                driver.save_screenshot(f"claude_2fa_{session.session_id}.png")
                return False
            except TimeoutException:
                pass  # No 2FA
            
            # Wait for main chat interface
            wait.until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Message Claude']")))
            logger.info(f"Claude login successful for {session.account_email}")
            
            # Save cookies
            with open(f"claude_cookies_{session.session_id}.pkl", "wb") as file:
                pickle.dump(driver.get_cookies(), file)
                
            return True
            
        except Exception as e:
            logger.error(f"Claude login failed: {str(e)}")
            return False
        
    async def _execute_claude_query(self, session: BrowserSession, prompt: str) -> str:
        """Execute Claude web interface automation"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 30)
            
            # Ensure on Claude chat
            if "claude.ai/chat" not in driver.current_url:
                driver.get("https://claude.ai/chat")
                
            # Find and type prompt
            textarea = wait.until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Message Claude']")))
            textarea.clear()
            for char in prompt:
                textarea.send_keys(char)
                await asyncio.sleep(0.05)
                
            # Submit
            textarea.send_keys(Keys.ENTER)
            
            # Capture streaming response
            full_response = ""
            last_response_length = 0
            retry_count = 0
            
            while retry_count < 5:
                try:
                    response_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.content p")))
                    current_response = response_div.text
                    
                    if len(current_response) > last_response_length:
                        full_response = current_response
                        last_response_length = len(full_response)
                        retry_count = 0
                    else:
                        retry_count += 1
                        
                    # Check for completion
                    if "●" not in full_response and "..." not in full_response:
                        break
                        
                    await asyncio.sleep(1)
                except:
                    retry_count += 1
                    await asyncio.sleep(1)
            
            # Update session stats
            session.usage_count += 1
            session.last_used = datetime.now()
            
            return full_response
            
        except Exception as e:
            logger.error(f"Claude query failed: {str(e)}")
            return ""
    
    async def _grok_login(self, session: BrowserSession):
        """Full Grok login flow with credential handling"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 20)
            
            # Navigate to Grok login
            driver.get("https://x.ai/")
            
            # Click login
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(.,'Sign in')]")))
            login_button.click()
            
            # Enter email
            email_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
            email_field.clear()
            email_field.send_keys(session.account_email)
            email_field.send_keys(Keys.RETURN)
            
            # Enter password
            password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
            password_field.clear()
            password_field.send_keys(session.account_password)
            password_field.send_keys(Keys.RETURN)
            
            # Handle 2FA if present
            try:
                twofa_field = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.NAME, "verificationCode"))
                )
                logger.warning("Grok 2FA detected - manual intervention required")
                driver.save_screenshot(f"grok_2fa_{session.session_id}.png")
                return False
            except TimeoutException:
                pass  # No 2FA
            
            # Wait for Grok chat interface
            wait.until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Message Grok']")))
            logger.info(f"Grok login successful for {session.account_email}")
            
            # Save cookies
            with open(f"grok_cookies_{session.session_id}.pkl", "wb") as file:
                pickle.dump(driver.get_cookies(), file)
                
            return True
            
        except Exception as e:
            logger.error(f"Grok login failed: {str(e)}")
            return False
        
    async def _execute_grok_query(self, session: BrowserSession, prompt: str) -> str:
        """Execute Grok web interface automation"""
        try:
            driver = session.driver
            wait = WebDriverWait(driver, 30)
            
            # Ensure on Grok chat
            if "x.ai" not in driver.current_url or "grok" not in driver.current_url:
                driver.get("https://x.ai/grok")
                
            # Find and type prompt
            textarea = wait.until(EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Message Grok']")))
            textarea.clear()
            for char in prompt:
                textarea.send_keys(char)
                await asyncio.sleep(0.05)
                
            # Submit
            textarea.send_keys(Keys.ENTER)
            
            # Capture streaming response
            full_response = ""
            last_response_length = 0
            retry_count = 0
            
            while retry_count < 5:
                try:
                    response_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.message-text")))
                    current_response = response_div.text
                    
                    if len(current_response) > last_response_length:
                        full_response = current_response
                        last_response_length = len(full_response)
                        retry_count = 0
                    else:
                        retry_count += 1
                        
                    # Check for completion
                    if "●" not in full_response and "..." not in full_response:
                        break
                        
                    await asyncio.sleep(1)
                except:
                    retry_count += 1
                    await asyncio.sleep(1)
            
            # Update session stats
            session.usage_count += 1
            session.last_used = datetime.now()
            
            return full_response
            
        except Exception as e:
            logger.error(f"Grok query failed: {str(e)}")
            return ""
    
    def _extract_json_response(self, full_response: str) -> Dict[str, Any]:
        """Robust JSON extraction with balanced brace scanning"""
        if not full_response:
            return {"signal": "HOLD", "confidence": 0.5, "reason": "EMPTY_RESPONSE"}

        cleaned = self._strip_code_fences(full_response)
        candidate = self._find_balanced_json(cleaned)
        
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                repaired = candidate.replace("'", '"')
                repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
                try:
                    return json.loads(repaired)
                except Exception:
                    logger.debug("JSON parse failed after repair")
        
        # Fallback: find any {...} via non-greedy regex
        for m in re.findall(r'\{[\s\S]*?\}', cleaned):
            try:
                return json.loads(m)
            except:
                m2 = m.replace("'", '"')
                m2 = re.sub(r',\s*([}\]])', r'\1', m2)
                try:
                    return json.loads(m2)
                except:
                    continue
        
        logger.warning("Failed to extract JSON from response")
        return {"signal": "HOLD", "confidence": 0.5, "reasoning_snippet": cleaned[:500]}
    
    def _strip_code_fences(self, text: str) -> str:
        """Remove Markdown code fences from JSON responses"""
        text = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text.strip()
    
    def _find_balanced_json(self, text: str) -> Optional[str]:
        """Find balanced braces for JSON extraction"""
        start = None
        depth = 0
        for i, ch in enumerate(text):
            if ch == '{':
                if start is None:
                    start = i
                depth += 1
            elif ch == '}':
                if start is not None:
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        return None
    
    def _normalize_signal(self, raw: Dict[str, Any], model: PremiumModelTier, symbol: str) -> Dict[str, Any]:
        """Validate and normalize signal structure"""
        def to_float(x, default=0.0):
            try:
                return float(x)
            except:
                return default
        
        signal = raw.get("signal", "HOLD")
        if isinstance(signal, str):
            signal = signal.upper()
            if signal not in ("BUY", "SELL", "HOLD"):
                signal = "HOLD"
        
        conf = to_float(raw.get("confidence", 0.5), 0.5)
        entry = to_float(raw.get("entry_price", None), None)
        stop = to_float(raw.get("stop_loss", None), None)
        tp = to_float(raw.get("take_profit", None), None)
        pos = to_float(raw.get("position_size", 0.01), 0.01)
        reasoning = raw.get("reasoning", raw.get("analysis", "")) or ""
        risk = raw.get("risk_level", "MEDIUM")
        timeframe_bias = raw.get("timeframe_bias", "NEUTRAL")

        return {
            "signal": signal,
            "confidence": max(0.0, min(conf, 1.0)),
            "entry_price": entry if entry is not None else 0.0,
            "stop_loss": stop if stop is not None else 0.0,
            "take_profit": tp if tp is not None else 0.0,
            "position_size": max(0.01, min(pos, 0.05)),
            "reasoning": reasoning,
            "risk_level": str(risk).upper(),
            "timeframe_bias": str(timeframe_bias).upper(),
            "model_used": model.value,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }

    async def _parse_signal_response(self, response: str, symbol: str, model: PremiumModelTier) -> Dict[str, Any]:
        """Parse and normalize signal response"""
        raw = self._extract_json_response(response)
        return self._normalize_signal(raw, model, symbol)

    async def _execute_chatgpt_query(self, session: BrowserSession, prompt: str) -> str:
        """Execute ChatGPT query in thread"""
        def _chatgpt_query_sync(driver, prompt):
            # Navigate to ChatGPT
            driver.get("https://chat.openai.com")
            
            # Wait for and find textarea
            textarea = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))
            
            # Clear and send prompt
            textarea.clear()
            textarea.send_keys(prompt)
            
            # Find and click send button
            send_button = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='send-button']")))
            send_button.click()
            
            # Wait for response
            response_div = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-message-author-role='assistant']")))
            
            # Extract response text
            response_text = response_div.text
            
            return response_text
        
        return await asyncio.to_thread(functools.partial(_chatgpt_query_sync, session.driver, prompt))
    
    async def _cleanup_oldest_session(self):
        """Clean up oldest session to free resources"""
        if not self.active_sessions:
            return
            
        oldest_session = min(self.active_sessions.values(), key=lambda s: s.last_used)
        await oldest_session.cleanup()
        
        # Remove from active sessions
        for provider, session in list(self.active_sessions.items()):
            if session == oldest_session:
                del self.active_sessions[provider]
                break
    
    async def cleanup(self):
        """Clean up all browser sessions"""
        try:
            for session in self.active_sessions.values():
                await session.cleanup()
            self.active_sessions.clear()
            logger.info("Premium Browser Engine cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            "is_initialized": self.is_initialized,
            "active_sessions": len(self.active_sessions),
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "available_providers": list(self.account_configs.keys()) if hasattr(self, 'account_configs') else [],
            "session_details": {
                provider: {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "usage_count": session.usage_count,
                    "success_rate": session.success_rate,
                    "last_used": session.last_used.isoformat()
                }
                for provider, session in self.active_sessions.items()
            },
            "model_hierarchy": {k: [m.value for m in v] for k, v in self.model_hierarchy.items()},
            "hardware_optimization": {
                "max_sessions": self.max_concurrent_sessions,
                "memory_limit_mb": self.session_memory_limit,
                "t470_optimized": True
            }
        }

    def _generate_fallback_signal(self, symbol: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback signal when premium models fail"""
        return {
            "signal": "HOLD",
            "confidence": 0.3,
            "entry_price": market_context.get("current_price", 0),
            "stop_loss": 0,
            "take_profit": 0,
            "position_size": 0.01,
            "reasoning": "Fallback signal - premium models unavailable",
            "risk_level": "LOW",
            "model_used": "fallback",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }
