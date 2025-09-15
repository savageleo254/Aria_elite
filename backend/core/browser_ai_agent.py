"""
ARIA ELITE Browser AI Agent
Headless browser automation for multi-AI integration (ChatGPT, Grok, Gemini, Claude)
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import random
import string

# Browser automation imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Alternative browser automation
try:
    import playwright
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class AIProvider(Enum):
    """Supported AI providers"""
    CHATGPT = "chatgpt"
    GROK = "grok"
    GEMINI = "gemini"
    CLAUDE = "claude"
    PERPLEXITY = "perplexity"

class AccountStatus(Enum):
    """Account status enumeration"""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    EXPIRED = "expired"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class AIAccount:
    """AI account configuration"""
    provider: AIProvider
    email: str
    password: str
    session_id: Optional[str] = None
    status: AccountStatus = AccountStatus.ACTIVE
    last_used: Optional[datetime] = None
    usage_count: int = 0
    daily_usage: int = 0
    rate_limit_reset: Optional[datetime] = None
    max_daily_usage: int = 100
    response_time_avg: float = 0.0
    error_count: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class BrowserAIManager:
    """
    Multi-AI browser automation manager
    Handles multiple AI service accounts through headless browsers
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.browser_instances = {}
        self.active_sessions = {}
        self.account_pool = {}
        self.provider_configs = {}
        self.rotation_strategy = "round_robin"  # round_robin, load_balanced, random
        self.max_concurrent_sessions = 5
        self.session_timeout = 1800  # 30 minutes
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'provider_stats': {provider.value: {
                'requests': 0,
                'success': 0,
                'failures': 0,
                'avg_time': 0.0
            } for provider in AIProvider}
        }
        
        self._load_provider_configs()
        
    async def initialize(self):
        """Initialize the browser AI manager"""
        try:
            logger.info("Initializing Browser AI Manager")
            
            # Check browser automation availability
            if not SELENIUM_AVAILABLE and not PLAYWRIGHT_AVAILABLE:
                raise Exception("No browser automation library available. Install selenium or playwright.")
            
            # Load account configurations
            await self._load_account_pool()
            
            # Initialize browser instances
            await self._initialize_browsers()
            
            # Start session cleanup task
            asyncio.create_task(self._session_cleanup_loop())
            
            self.is_initialized = True
            logger.info(f"Browser AI Manager initialized with {len(self.account_pool)} accounts")
            
        except Exception as e:
            logger.error(f"Failed to initialize Browser AI Manager: {str(e)}")
            raise
    
    def _load_provider_configs(self):
        """Load provider-specific configurations"""
        self.provider_configs = {
            AIProvider.CHATGPT: {
                'base_url': 'https://chat.openai.com',
                'login_url': 'https://chat.openai.com/auth/login',
                'api_endpoint': '/api/chat/completions',
                'max_tokens': 4000,
                'timeout': 30,
                'headless': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            AIProvider.GROK: {
                'base_url': 'https://grok.x.ai',
                'login_url': 'https://grok.x.ai/login',
                'api_endpoint': '/api/chat',
                'max_tokens': 8000,
                'timeout': 45,
                'headless': True,
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            },
            AIProvider.GEMINI: {
                'base_url': 'https://gemini.google.com',
                'login_url': 'https://gemini.google.com/app/login',
                'api_endpoint': '/api/v1/chat',
                'max_tokens': 32000,
                'timeout': 60,
                'headless': True,
                'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            },
            AIProvider.CLAUDE: {
                'base_url': 'https://claude.ai',
                'login_url': 'https://claude.ai/login',
                'api_endpoint': '/api/complete',
                'max_tokens': 100000,
                'timeout': 90,
                'headless': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0'
            },
            AIProvider.PERPLEXITY: {
                'base_url': 'https://perplexity.ai',
                'login_url': 'https://perplexity.com/login',
                'api_endpoint': '/api/chat/completions',
                'max_tokens': 16000,
                'timeout': 30,
                'headless': True,
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        }
    
    async def _load_account_pool(self):
        """Load AI account configurations"""
        try:
            # Load from configuration
            ai_config = self.config.load_ai_accounts_config()
            
            for account_data in ai_config.get('accounts', []):
                account = AIAccount(
                    provider=AIProvider(account_data['provider']),
                    email=account_data['email'],
                    password=account_data['password'],
                    max_daily_usage=account_data.get('max_daily_usage', 100),
                    session_id=account_data.get('session_id')
                )
                
                if account.provider.value not in self.account_pool:
                    self.account_pool[account.provider.value] = []
                
                self.account_pool[account.provider.value].append(account)
                logger.info(f"Loaded {account.provider.value} account: {account.email}")
            
            # Generate mock accounts if none provided
            if not self.account_pool:
                logger.warning("No AI accounts configured, generating mock accounts for testing")
                await self._generate_mock_accounts()
                
        except Exception as e:
            logger.error(f"Failed to load account pool: {str(e)}")
            await self._generate_mock_accounts()
    
    async def _generate_mock_accounts(self):
        """Generate mock accounts for testing"""
        mock_accounts = [
            {
                'provider': 'chatgpt',
                'email': 'test@example.com',
                'password': 'test_password',
                'max_daily_usage': 50
            },
            {
                'provider': 'gemini',
                'email': 'test@example.com',
                'password': 'test_password',
                'max_daily_usage': 100
            },
            {
                'provider': 'claude',
                'email': 'test@example.com',
                'password': 'test_password',
                'max_daily_usage': 75
            }
        ]
        
        for account_data in mock_accounts:
            account = AIAccount(
                provider=AIProvider(account_data['provider']),
                email=account_data['email'],
                password=account_data['password'],
                max_daily_usage=account_data['max_daily_usage']
            )
            
            if account.provider.value not in self.account_pool:
                self.account_pool[account.provider.value] = []
            
            self.account_pool[account.provider.value].append(account)
    
    async def _initialize_browsers(self):
        """Initialize browser instances"""
        if PLAYWRIGHT_AVAILABLE:
            await self._initialize_playwright_browsers()
        elif SELENIUM_AVAILABLE:
            await self._initialize_selenium_browsers()
        else:
            raise Exception("No browser automation available")
    
    async def _initialize_playwright_browsers(self):
        """Initialize Playwright browsers"""
        try:
            self.playwright = await async_playwright().start()
            
            for provider in AIProvider:
                config = self.provider_configs[provider]
                
                browser = await self.playwright.chromium.launch(
                    headless=config['headless'],
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-extensions',
                        '--disable-plugins',
                        '--disable-images',  # Disable images for performance
                        '--disable-javascript-harmony-blocking',
                        '--disable-web-security'
                    ]
                )
                
                self.browser_instances[provider.value] = browser
                logger.info(f"Initialized Playwright browser for {provider.value}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Playwright browsers: {str(e)}")
            raise
    
    async def _initialize_selenium_browsers(self):
        """Initialize Selenium browsers"""
        try:
            for provider in AIProvider:
                config = self.provider_configs[provider]
                
                options = Options()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--disable-extensions')
                options.add_argument('--disable-plugins')
                options.add_argument('--disable-images')
                options.add_argument(f'user-agent={config["user_agent"]}')
                
                service = Service('/usr/bin/chromedriver')  # Update path as needed
                
                browser = webdriver.Chrome(service=service, options=options)
                self.browser_instances[provider.value] = browser
                
                logger.info(f"Initialized Selenium browser for {provider.value}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Selenium browsers: {str(e)}")
            raise
    
    async def get_best_account(self, provider: AIProvider) -> Optional[AIAccount]:
        """Get the best available account for a provider"""
        provider_accounts = self.account_pool.get(provider.value, [])
        
        if not provider_accounts:
            logger.warning(f"No accounts available for {provider.value}")
            return None
        
        # Filter accounts by status
        available_accounts = [
            acc for acc in provider_accounts 
            if acc.status == AccountStatus.ACTIVE 
            and (acc.rate_limit_reset is None or acc.rate_limit_reset < datetime.now())
            and acc.daily_usage < acc.max_daily_usage
        ]
        
        if not available_accounts:
            logger.warning(f"No available accounts for {provider.value}")
            return None
        
        # Apply rotation strategy
        if self.rotation_strategy == "round_robin":
            return available_accounts[0]
        elif self.rotation_strategy == "load_balanced":
            return min(available_accounts, key=lambda x: x.usage_count)
        elif self.rotation_strategy == "random":
            return random.choice(available_accounts)
        else:
            return available_accounts[0]
    
    async def generate_signal(self, symbol: str, timeframe: str, strategy: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate trading signal using multi-AI system
        """
        try:
            logger.info(f"Generating signal for {symbol} using multi-AI system")
            
            # Prepare prompt for AI models
            prompt = self._prepare_trading_prompt(symbol, timeframe, strategy, parameters)
            
            # Try different AI providers in order of preference
            providers_to_try = [
                AIProvider.CHATGPT,
                AIProvider.GEMINI,
                AIProvider.CLAUDE,
                AIProvider.GROK,
                AIProvider.PERPLEXITY
            ]
            
            for provider in providers_to_try:
                try:
                    account = await self.get_best_account(provider)
                    if not account:
                        continue
                    
                    logger.info(f"Trying {provider.value} with account {account.email}")
                    
                    # Generate AI response
                    response = await self._get_ai_response(provider, account, prompt)
                    
                    if response:
                        # Parse and format the response
                        signal = await self._parse_ai_response(response, symbol, strategy)
                        
                        # Update account stats
                        account.usage_count += 1
                        account.daily_usage += 1
                        account.last_used = datetime.now()
                        
                        # Update performance stats
                        self.performance_stats['total_requests'] += 1
                        self.performance_stats['successful_requests'] += 1
                        
                        logger.info(f"✅ Signal generated using {provider.value}")
                        return signal
                        
                except Exception as e:
                    logger.error(f"❌ {provider.value} failed: {str(e)}")
                    self.performance_stats['failed_requests'] += 1
                    
                    # Mark account as error if it fails
                    if account:
                        account.status = AccountStatus.ERROR
                        account.error_count += 1
            
            # If all providers fail, return a default signal
            logger.warning("All AI providers failed, returning default signal")
            return await self._generate_default_signal(symbol, strategy)
            
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            raise
    
    async def _get_ai_response(self, provider: AIProvider, account: AIAccount, prompt: str) -> Optional[str]:
        """Get AI response from specific provider"""
        try:
            config = self.provider_configs[provider]
            
            # Login if needed
            if not account.session_id:
                await self._login_to_provider(provider, account)
            
            # Send prompt to AI
            if PLAYWRIGHT_AVAILABLE:
                return await self._get_playwright_response(provider, account, prompt)
            elif SELENIUM_AVAILABLE:
                return await self._get_selenium_response(provider, account, prompt)
            else:
                raise Exception("No browser automation available")
                
        except Exception as e:
            logger.error(f"Failed to get response from {provider.value}: {str(e)}")
            return None
    
    async def _login_to_provider(self, provider: AIProvider, account: AIAccount):
        """Login to AI provider"""
        try:
            logger.info(f"Logging into {provider.value} with {account.email}")
            
            config = self.provider_configs[provider]
            
            if PLAYWRIGHT_AVAILABLE:
                await self._playwright_login(provider, account)
            elif SELENIUM_AVAILABLE:
                await self._selenium_login(provider, account)
            
            account.session_id = self._generate_session_id()
            account.status = AccountStatus.ACTIVE
            
            logger.info(f"✅ Successfully logged into {provider.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to login to {provider.value}: {str(e)}")
            account.status = AccountStatus.ERROR
            raise
    
    async def _playwright_login(self, provider: AIProvider, account: AIAccount):
        """Login using Playwright"""
        browser = self.browser_instances[provider.value]
        
        try:
            page = await browser.new_page()
            
            # Navigate to login page
            await page.goto(config['login_url'])
            
            # Fill login form
            await page.fill('input[type="email"]', account.email)
            await page.fill('input[type="password"]', account.password)
            await page.click('button[type="submit"]')
            
            # Wait for login to complete
            await page.wait_for_load_state('networkidle')
            
            # Save session cookies
            cookies = await page.context.cookies()
            # Store cookies for future use
            
            await page.close()
            
        except Exception as e:
            logger.error(f"Playwright login failed: {str(e)}")
            raise
    
    async def _selenium_login(self, provider: AIProvider, account: AIAccount):
        """Login using Selenium"""
        browser = self.browser_instances[provider.value]
        
        try:
            # Navigate to login page
            browser.get(config['login_url'])
            
            # Wait for login form to load
            WebDriverWait(browser, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="email"]'))
            )
            
            # Fill login form
            email_input = browser.find_element(By.CSS_SELECTOR, 'input[type="email"]')
            password_input = browser.find_element(By.CSS_SELECTOR, 'input[type="password"]')
            submit_button = browser.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
            
            email_input.clear()
            email_input.send_keys(account.email)
            password_input.clear()
            password_input.send_keys(account.password)
            submit_button.click()
            
            # Wait for login to complete
            time.sleep(10)  # Simple wait, could be improved
            
        except Exception as e:
            logger.error(f"Selenium login failed: {str(e)}")
            raise
    
    async def _get_playwright_response(self, provider: AIProvider, account: AIAccount, prompt: str) -> str:
        """Get AI response using Playwright"""
        browser = self.browser_instances[provider.value]
        config = self.provider_configs[provider]
        
        try:
            page = await browser.new_page()
            
            # Navigate to chat interface
            await page.goto(config['base_url'])
            
            # Fill prompt
            await page.fill('textarea[placeholder*="Ask" i], textarea[placeholder*="Message" i]', prompt)
            await page.click('button[type="submit"], button:has-text("Send")')
            
            # Wait for response
            await page.wait_for_selector('.response, .message', timeout=config['timeout'] * 1000)
            
            # Extract response
            response_element = await page.query_selector('.response, .message')
            response = await response_element.inner_text()
            
            await page.close()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Playwright response failed: {str(e)}")
            raise
    
    async def _get_selenium_response(self, provider: AIProvider, account: AIAccount, prompt: str) -> str:
        """Get AI response using Selenium"""
        browser = self.browser_instances[provider.value]
        config = self.provider_configs[provider]
        
        try:
            # Navigate to chat interface
            browser.get(config['base_url'])
            
            # Wait for chat interface
            WebDriverWait(browser, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'textarea[placeholder*="Ask" i], textarea[placeholder*="Message" i]'))
            )
            
            # Fill prompt
            prompt_input = browser.find_element(By.CSS_SELECTOR, 'textarea[placeholder*="Ask" i], textarea[placeholder*="Message" i]')
            send_button = browser.find_element(By.CSS_SELECTOR, 'button[type="submit"], button:has-text("Send")')
            
            prompt_input.clear()
            prompt_input.send_keys(prompt)
            send_button.click()
            
            # Wait for response
            time.sleep(config['timeout'])
            
            # Extract response
            response_elements = browser.find_elements(By.CSS_SELECTOR, '.response, .message, .answer')
            response = "\n".join([el.text for el in response_elements if el.text.strip()])
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Selenium response failed: {str(e)}")
            raise
    
    def _prepare_trading_prompt(self, symbol: str, timeframe: str, strategy: str, parameters: Dict[str, Any] = None) -> str:
        """Prepare trading prompt for AI models"""
        base_prompt = f"""
You are an expert trading AI assistant. Analyze the following trading request and provide a comprehensive trading signal.

TRADING REQUEST:
- Symbol: {symbol}
- Timeframe: {timeframe}
- Strategy: {strategy}
- Parameters: {json.dumps(parameters or {}, indent=2)}

Please provide a detailed analysis including:
1. Market analysis and current conditions
2. Technical indicators assessment
3. Risk analysis and potential scenarios
4. Specific entry/exit recommendations
5. Stop loss and take profit levels
6. Confidence score (0.0-1.0)

Format your response as JSON:
{{
  "analysis": "Detailed market analysis",
  "signal_direction": "buy|sell|hold",
  "entry_price": 1.1000,
  "stop_loss": 1.0950,
  "take_profit": 1.1050,
  "confidence": 0.75,
  "risk_level": "low|medium|high",
  "time_horizon": "short|medium|long",
  "key_factors": ["factor1", "factor2"],
  "recommendation": "Detailed recommendation"
}}
"""
        return base_prompt
    
    async def _parse_ai_response(self, response: str, symbol: str, strategy: str) -> Dict[str, Any]:
        """Parse AI response and format as trading signal"""
        try:
            # Try to parse JSON response
            data = json.loads(response)
            
            return {
                "signal_id": f"ai_{symbol}_{strategy}_{int(time.time())}",
                "symbol": symbol,
                "direction": data.get("signal_direction", "hold"),
                "entry_price": float(data.get("entry_price", 0)),
                "stop_loss": float(data.get("stop_loss", 0)),
                "take_profit": float(data.get("take_profit", 0)),
                "confidence": float(data.get("confidence", 0.5)),
                "strategy": strategy,
                "timeframe": "1h",
                "analysis": data.get("analysis", ""),
                "risk_level": data.get("risk_level", "medium"),
                "time_horizon": data.get("time_horizon", "medium"),
                "key_factors": data.get("key_factors", []),
                "recommendation": data.get("recommendation", ""),
                "ai_provider": "multi_ai",
                "generated_at": datetime.now().isoformat()
            }
            
        except json.JSONDecodeError:
            # Fallback parsing for non-JSON responses
            logger.warning("AI response not in JSON format, using fallback parsing")
            
            return {
                "signal_id": f"ai_{symbol}_{strategy}_{int(time.time())}",
                "symbol": symbol,
                "direction": "hold",  # Default to hold
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "confidence": 0.5,
                "strategy": strategy,
                "timeframe": "1h",
                "analysis": response,
                "risk_level": "medium",
                "time_horizon": "medium",
                "key_factors": [],
                "recommendation": "AI response received but not properly formatted",
                "ai_provider": "multi_ai",
                "generated_at": datetime.now().isoformat()
            }
    
    async def _generate_default_signal(self, symbol: str, strategy: str) -> Dict[str, Any]:
        """Generate default signal when AI providers fail"""
        logger.warning(f"Generating default signal for {symbol}")
        
        return {
            "signal_id": f"default_{symbol}_{strategy}_{int(time.time())}",
            "symbol": symbol,
            "direction": "hold",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "confidence": 0.1,
            "strategy": strategy,
            "timeframe": "1h",
            "analysis": "Default signal - all AI providers unavailable",
            "risk_level": "high",
            "time_horizon": "short",
            "key_factors": ["ai_unavailable"],
            "recommendation": "No trading recommendation available due to AI service outage",
            "ai_provider": "default",
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{int(time.time())}_{random.randint(1000, 9999)}"
    
    async def _session_cleanup_loop(self):
        """Clean up expired sessions"""
        while True:
            try:
                now = datetime.now()
                
                # Clean up expired sessions
                expired_sessions = []
                for session_id, session_data in self.active_sessions.items():
                    if session_data['expires_at'] < now:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    logger.info(f"Cleaned up expired session: {session_id}")
                
                # Clean up error accounts
                for provider, accounts in self.account_pool.items():
                    for account in accounts:
                        if account.status == AccountStatus.ERROR and account.error_count > 3:
                            logger.warning(f"Account {account.email} has too many errors, marking as expired")
                            account.status = AccountStatus.EXPIRED
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Session cleanup error: {str(e)}")
                await asyncio.sleep(300)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        total_accounts = sum(len(accounts) for accounts in self.account_pool.values())
        active_accounts = sum(
            sum(1 for acc in accounts if acc.status == AccountStatus.ACTIVE)
            for accounts in self.account_pool.values()
        )
        
        return {
            "browser_ai_manager": {
                "initialized": self.is_initialized,
                "total_accounts": total_accounts,
                "active_accounts": active_accounts,
                "active_sessions": len(self.active_sessions),
                "rotation_strategy": self.rotation_strategy,
                "max_concurrent_sessions": self.max_concurrent_sessions,
                "performance_stats": self.performance_stats,
                "providers_status": {
                    provider.value: {
                        "accounts": len(self.account_pool.get(provider.value, [])),
                        "active": sum(1 for acc in self.account_pool.get(provider.value, []) if acc.status == AccountStatus.ACTIVE),
                        "last_request": acc.last_used.isoformat() if self.account_pool.get(provider.value, [{}])[0].last_used else None
                    }
                    for provider in AIProvider
                }
            }
        }
    
    async def add_account(self, provider: AIProvider, email: str, password: str, max_daily_usage: int = 100) -> bool:
        """Add new AI account"""
        try:
            account = AIAccount(
                provider=provider,
                email=email,
                password=password,
                max_daily_usage=max_daily_usage
            )
            
            if provider.value not in self.account_pool:
                self.account_pool[provider.value] = []
            
            self.account_pool[provider.value].append(account)
            
            logger.info(f"Added {provider.value} account: {email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add account: {str(e)}")
            return False
    
    async def remove_account(self, provider: AIProvider, email: str) -> bool:
        """Remove AI account"""
        try:
            if provider.value in self.account_pool:
                self.account_pool[provider.value] = [
                    acc for acc in self.account_pool[provider.value] 
                    if acc.email != email
                ]
                
                logger.info(f"Removed {provider.value} account: {email}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove account: {str(e)}")
            return False
    
    async def close(self):
        """Close all browser instances"""
        try:
            if PLAYWRIGHT_AVAILABLE:
                await self.playwright.stop()
            
            if SELENIUM_AVAILABLE:
                for browser in self.browser_instances.values():
                    browser.quit()
            
            logger.info("Browser AI Manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Browser AI Manager: {str(e)}")

# Global instance
browser_ai_manager = BrowserAIManager()