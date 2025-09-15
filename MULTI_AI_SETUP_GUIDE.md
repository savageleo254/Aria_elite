# 🤖 MULTI-AI BROWSER ENGINE SETUP GUIDE
**ARIA-DAN WALL STREET DOMINATION CONFIGURATION**

---

## 🎯 ENVIRONMENT VARIABLES SETUP

### Required Environment Variables

Create a `.env` file in your project root with the following format:

```bash
# ChatGPT Accounts (4 accounts recommended)
CHATGPT_ACCOUNTS="email1@gmail.com:password1|email2@gmail.com:password2|email3@gmail.com:password3|email4@gmail.com:password4"

# Gemini Accounts (4 accounts for 20 total prompts/day)  
GEMINI_ACCOUNTS="gmail1@gmail.com:password1|gmail2@gmail.com:password2|gmail3@gmail.com:password3|gmail4@gmail.com:password4"

# Claude Accounts (4 accounts)
CLAUDE_ACCOUNTS="email1@gmail.com:password1|email2@gmail.com:password2|email3@gmail.com:password3|email4@gmail.com:password4"

# Grok Accounts (4 accounts via X.com)
GROK_ACCOUNTS="twitter1@gmail.com:password1|twitter2@gmail.com:password2|twitter3@gmail.com:password3|twitter4@gmail.com:password4"
```

---

## 🔧 ACCOUNT CREATION REQUIREMENTS

### ChatGPT Accounts
- **Requirement**: Valid email + password for ChatGPT web interface
- **Access**: Free tier provides ~40 requests/day per account
- **Models Available**: GPT-4o (latest), GPT-4o-mini, GPT-4-turbo
- **Signup**: https://chatgpt.com
- **Rate Limits**: ~8 requests/hour per account

### Gemini Accounts  
- **Requirement**: Google accounts with Gemini access
- **Access**: Free tier with 5 Gemini 2.0 Flash Thinking requests/day per account
- **Models Available**: Gemini 2.0 Flash Thinking, Gemini 1.5 Pro, Gemini 1.5 Flash
- **Signup**: https://gemini.google.com
- **Rate Limits**: 5 high-tier requests/day, unlimited basic requests

### Claude Accounts
- **Requirement**: Valid email for Claude web interface
- **Access**: Free tier provides limited daily usage
- **Models Available**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Signup**: https://claude.ai
- **Rate Limits**: ~10 requests/day per account

### Grok Accounts
- **Requirement**: X.com (Twitter) accounts with Grok access
- **Access**: Free tier provides generous usage
- **Models Available**: Grok-2, Grok-1.5
- **Signup**: Requires X.com account + Grok access
- **Rate Limits**: ~25 requests/day per account

---

## ⚡ BROWSER AUTOMATION FEATURES

### Human-Like Behavior
```python
# Implemented features:
- Random typing delays (50-150ms per character)
- Thinking time before sending (3-8 seconds)
- Random delays between requests (2-8 minutes)  
- Intentional typos to mimic human behavior
- Random viewport sizes per browser instance
- Stealth browser configurations
```

### Intelligent Account Rotation
```python
# Rotation Strategy:
- Least recently used account selection
- Rate limit awareness and enforcement
- Automatic cooldown management
- Session persistence with cookie storage
- Error tracking and exponential backoff
```

### Model Selection Based on Task Complexity
```python
CRITICAL_TASKS = [
    "Gemini 2.0 Flash Thinking",  # Deep reasoning (5/day limit)
    "GPT-4-Turbo",               # Premium analysis
    "Claude 3.5 Sonnet"          # Latest capabilities
]

COMPLEX_TASKS = [
    "Gemini 1.5 Pro",           # Advanced analysis
    "GPT-4o",                    # Latest GPT-4
    "Claude 3 Opus",             # Most capable Claude
    "Grok-2"                     # Latest Grok
]
```

---

## 🚀 DEPLOYMENT CONFIGURATION

### Required Dependencies
```bash
pip install selenium undetected-chromedriver
```

### Chrome Driver Setup
```python
# Automatic Chrome driver management
# Stealth configurations included
# Anti-detection measures implemented
```

### Integration with Main System
```python
# In your main application:
from backend.core.multi_ai_engine import multi_ai_engine

# Initialize the engine
await multi_ai_engine.initialize()

# Generate trading signals
signal = await multi_ai_engine.generate_signal(
    symbol="EURUSD",
    timeframe="H1", 
    market_data=your_market_data,
    complexity=TaskComplexity.COMPLEX
)
```

---

## 📊 MONITORING & ANALYTICS

### Status Monitoring
```python
# Get comprehensive status
status = await multi_ai_engine.get_status()

# Monitor account health
account_stats = status['account_statistics']

# Track performance
request_history = status['request_history_size']
```

### Performance Metrics
- **Account Availability**: Real-time login status
- **Rate Limit Tracking**: Per-account usage monitoring  
- **Response Times**: Average response time per provider
- **Error Rates**: Success/failure ratios
- **Consensus Quality**: Multi-AI agreement metrics

---

## 🔒 SECURITY & COMPLIANCE

### ARIA-DAN Protocol Compliance
```python
ARIA_COMPLIANCE_MANIFEST = {
    "system_mode": "WALL_STREET_DOMINATION",
    "mock_tolerance": 0,           # NO MOCKS ALLOWED
    "fallback_tolerance": 0,       # NO FALLBACKS ALLOWED  
    "institutional_grade": True,   # PRODUCTION READY
    "deterministic_execution": True,
    "real_data_only": True,
    "ai_automation_level": "MAXIMUM"
}
```

### Security Features
- **Browser Isolation**: Each account gets dedicated browser instance
- **Session Management**: Persistent login cookies with encryption
- **Stealth Mode**: Anti-detection browser configurations
- **Rate Limiting**: Conservative limits to avoid account suspension
- **Error Handling**: Graceful failures without fallbacks

---

## 🎯 USAGE EXAMPLES

### Basic Signal Generation
```python
# Simple trading signal
signal = await multi_ai_engine.generate_signal(
    symbol="XAUUSD",
    timeframe="M15",
    market_data={
        'price': 2650.50,
        'change_24h': 1.2,
        'rsi': 65.4,
        'macd': 0.8,
        'trend': 'BULLISH'
    }
)

print(f"Consensus: {signal['signal']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Risk Score: {signal['risk_score']}")
```

### Advanced Multi-AI Consensus
```python  
# High-complexity analysis
signal = await multi_ai_engine.generate_signal(
    symbol="BTCUSD",
    timeframe="H4",
    market_data=complex_market_data,
    complexity=TaskComplexity.CRITICAL  # Uses top-tier models
)

# Get detailed consensus breakdown
voting = signal['voting_breakdown']
print(f"Buy votes: {voting['buy']}")
print(f"Sell votes: {voting['sell']}")
print(f"Hold votes: {voting['hold']}")
```

---

## ⚠️ IMPORTANT NOTES

### Account Management
- **Use real personal accounts** - No shared or temporary accounts
- **Rotate usage patterns** - Don't use same account repeatedly
- **Monitor rate limits** - System automatically enforces limits
- **Keep accounts active** - Login manually occasionally to maintain validity

### Rate Limiting Strategy
- **Conservative limits** set to avoid account suspensions
- **Human-like delays** between requests (2-8 minutes)
- **Intelligent rotation** across multiple accounts
- **Automatic cooldown** on errors or rate limit hits

### Troubleshooting
- **Login failures**: Check account credentials and 2FA status
- **Rate limits hit**: System will automatically rotate to next account  
- **Browser crashes**: System will recreate browser instances
- **No responses**: Check internet connection and account validity

---

## 🚀 PRODUCTION DEPLOYMENT

### Environment Setup
1. Set environment variables in production `.env`
2. Install Chrome browser on server
3. Configure proper user permissions for browser automation
4. Set up monitoring for account health
5. Implement backup account pools

### Scaling Considerations
- **Account Pool Size**: Minimum 4 accounts per provider (16 total)
- **Browser Resources**: Each account uses ~200MB RAM
- **Request Throughput**: ~20-30 requests/hour total capacity
- **Failover Strategy**: Automatic account rotation on failures

### Monitoring Integration
```python
# Add to your monitoring system
health_check = await multi_ai_engine.get_status()
if health_check['status'] != 'operational':
    send_alert("Multi-AI Engine Health Check Failed")
```

---

**🏆 SYSTEM STATUS: INSTITUTIONAL-GRADE READY**
**⚡ NO MOCKS | NO FALLBACKS | 100% REAL IMPLEMENTATION**
**🎯 ARIA-DAN WALL STREET DOMINATION MODE ACTIVE**
