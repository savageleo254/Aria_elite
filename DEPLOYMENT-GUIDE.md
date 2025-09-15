# ARIA ELITE Deployment Guide

**Production Deployment for Institutional Trading**

This guide covers the complete deployment process for ARIA ELITE trading system in production environments.

## 🚨 Pre-Deployment Checklist

### ✅ System Requirements
- **Python 3.9+** with pip and virtual environment support
- **Node.js 18+** with npm package manager
- **MetaTrader 5** account with API access enabled
- **Google Gemini API** key with sufficient quota
- **Discord Bot** (optional but recommended for notifications)
- **Windows 10/11** or **Linux Ubuntu 20.04+**

### ✅ Account Prerequisites
1. **MT5 Trading Account**:
   - Live or demo account with API access
   - Account credentials (login, password, server)
   - Verify connection via MT5 terminal

2. **Google Gemini API**:
   - API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Verify quota limits and billing setup

3. **Discord Integration** (Optional):
   - Discord bot token from [Discord Developer Portal](https://discord.com/developers/applications)
   - Webhook URL for notifications
   - User ID for authorized access

### 3. AI Provider Account Setup

**CRITICAL: You MUST create dedicated accounts using EMAIL/PASSWORD authentication.** Google Sign-In (OAuth) is NOT supported.

#### Step-by-Step Account Creation:
1. **ChatGPT Plus**
   - Visit: https://chat.openai.com
   - Click "Sign up" → Use email address (NOT "Continue with Google")
   - Complete payment for ChatGPT Plus subscription ($20/month)

2. **Gemini Advanced**
   - Visit: https://gemini.google.com
   - Click "Sign in" → "Create account" → "For myself"
   - Use NEW email (not your primary Gmail)
   - Complete Google One subscription ($20/month)

3. **Claude Pro**
   - Visit: https://claude.ai
   - Click "Sign up with email"
   - Use NEW email address
   - Complete Pro subscription ($20/month)

4. **Grok Premium**
   - Visit: https://x.ai
   - Click "Sign up" → Use NEW email
   - Complete Premium subscription ($16/month)

#### Security Recommendations:
- Use password manager (Bitwarden/1Password)
- Generate unique 20+ character passwords
- Use non-Gmail addresses (ProtonMail, Outlook, etc.)
- **Disable 2FA** during initial setup (will implement automation later)

## 🔧 Environment Setup

### 1. System Preparation

**Windows:**
```cmd
# Install Python 3.9+
winget install Python.Python.3.9

# Install Node.js 18+
winget install OpenJS.NodeJS

# Verify installations
python --version
node --version
npm --version
```

**Linux:**
```bash
# Install Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

### 2. Project Setup

```bash
# Clone repository
git clone <repository-url>
cd ARIA-ELITE

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.prod.txt

# Install Node.js dependencies
npm install

# Initialize database
npm run setup
```

### 3. Environment Configuration

```bash
# Copy environment template
cp production.env.example .env

# Edit .env file with your credentials
nano .env  # Linux
notepad .env  # Windows
```

**Required Environment Variables:**
```bash
# Core System
NODE_ENV=production
DATABASE_URL=file:./db/custom.db

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Security Tokens (generate with: openssl rand -base64 32)
SUPERVISOR_API_TOKEN=your_secure_supervisor_token
JWT_SECRET=your_secure_jwt_secret
ENCRYPTION_KEY=your_secure_encryption_key

# MetaTrader 5 Connection
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# Discord Notifications (Optional)
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_WEBHOOK_URL=your_discord_webhook_url
DISCORD_ALLOWED_USERS=user_id_1,user_id_2

# System Configuration
LOG_LEVEL=info
MAX_WORKERS=4
WORKER_TIMEOUT=30
```

## 🚀 Deployment Methods

### Method 1: Direct Deployment

**1. Build Frontend:**
```bash
npm run build
```

**2. Start Backend API:**
```bash
# Production mode
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**3. Start Frontend:**
```bash
npm run start
```

**4. Start Trading System:**
```bash
python start_live_trading.py
```

### Method 2: Docker Deployment

**1. Build Docker Images:**
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build
```

**2. Deploy Services:**
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d
```

**3. Verify Deployment:**
```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Method 3: Windows Service Deployment

**1. Install Python Service Wrapper:**
```cmd
pip install pywin32
```

**2. Create Service Configuration:**
```python
# service_config.py
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import subprocess
import os

class ARIAELITEService(win32serviceutil.ServiceFramework):
    _svc_name_ = "ARIA_ELITE_Trading"
    _svc_display_name_ = "ARIA ELITE Trading System"
    _svc_description_ = "AI-powered institutional trading system"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.main()

    def main(self):
        # Start the trading system
        subprocess.run([
            "python", "start_live_trading.py"
        ], cwd=r"C:\path\to\ARIA-ELITE")

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(ARIAELITEService)
```

**3. Install and Start Service:**
```cmd
python service_config.py install
python service_config.py start
```

## 🔍 Post-Deployment Verification

### 1. System Health Checks

**API Health:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "timestamp": "..."}
```

**System Status:**
```bash
curl -H "Authorization: Bearer your_token" http://localhost:8000/status
# Expected: Complete system status with metrics
```

**Frontend Access:**
- Navigate to: http://localhost:3000
- Verify dashboard loads correctly
- Check real-time data updates

### 2. Trading System Verification

**Signal Generation Test:**
```bash
curl -X POST http://localhost:8000/signals/generate \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "1h", 
    "strategy": "smc_strategy"
  }'
```

**MT5 Connection Test:**
```python
# test_mt5_connection.py
from backend.core.mt5_bridge import MT5Bridge
import asyncio

async def test_connection():
    bridge = MT5Bridge()
    await bridge.initialize()
    
    if bridge.is_connected():
        print("✅ MT5 connection successful")
        account_info = await bridge.get_account_info()
        print(f"Account: {account_info.get('login')}")
    else:
        print("❌ MT5 connection failed")

asyncio.run(test_connection())
```

### 3. Security Verification

**Token Authentication:**
```bash
# Test without token (should fail)
curl http://localhost:8000/status
# Expected: 401 Unauthorized

# Test with valid token (should succeed)
curl -H "Authorization: Bearer your_token" http://localhost:8000/status
# Expected: 200 OK with status data
```

**Environment Security:**
```bash
# Verify no secrets in logs
grep -r "GEMINI_API_KEY\|MT5_PASSWORD" logs/
# Should return no results

# Check file permissions
ls -la .env
# Should be readable only by owner (600)
```

## 🔧 Configuration Management

### Trading Configuration

**Strategy Parameters** (`configs/strategy_config.json`):
```json
{
  "strategies": {
    "smc_strategy": {
      "enabled": true,
      "confidence_threshold": 0.75,
      "risk_per_trade": 0.02,
      "max_concurrent_trades": 3
    },
    "ai_strategy": {
      "enabled": true,
      "model_weights": {
        "lightgbm": 0.4,
        "cnn": 0.3,
        "mobilenet": 0.3
      }
    }
  }
}
```

**Execution Settings** (`configs/execution_config.json`):
```json
{
  "risk_management": {
    "max_daily_loss": 0.05,
    "max_drawdown": 0.15,
    "position_sizing": "atr_based",
    "stop_loss_multiplier": 2.0,
    "take_profit_multiplier": 3.0
  },
  "execution": {
    "slippage_tolerance": 0.0005,
    "retry_attempts": 3,
    "timeout_seconds": 30
  }
}
```

### System Monitoring

**Log Configuration:**
```bash
# Rotate logs daily, keep 30 days
LOG_ROTATION=daily
LOG_RETENTION_DAYS=30
LOG_MAX_SIZE=100MB
```

**Alert Thresholds:**
```json
{
  "alerts": {
    "critical_loss_threshold": -0.03,
    "connection_timeout": 30,
    "memory_usage_threshold": 0.85,
    "cpu_usage_threshold": 0.90
  }
}
```

## 🚨 Emergency Procedures

### Kill Switch Activation

**Manual Kill Switch:**
```bash
# Via API
curl -X POST http://localhost:8000/system/kill-switch \
  -H "Authorization: Bearer your_token"

# Via Discord (if configured)
# In Discord: !kill-switch

# Direct Python
python -c "
import asyncio
from backend.core.execution_engine import ExecutionEngine
async def emergency_stop():
    engine = ExecutionEngine()
    await engine.initialize()
    await engine.emergency_close_all()
asyncio.run(emergency_stop())
"
```

**Automated Kill Switch Triggers:**
- Daily loss exceeds 5%
- Maximum drawdown reaches 15% 
- API connection lost for >5 minutes
- Critical system errors detected

### System Recovery

**Recovery Steps:**
1. **Stop all trading operations**
2. **Verify account status and positions**
3. **Check system logs for errors**
4. **Restart system components**
5. **Validate all connections**
6. **Resume trading with reduced risk**

**Recovery Script:**
```bash
#!/bin/bash
# recovery.sh

echo "🚨 Starting system recovery..."

# Stop trading system
pkill -f "start_live_trading.py"

# Wait for clean shutdown
sleep 10

# Check for stuck processes
ps aux | grep -E "(python|node)" | grep -v grep

# Restart system
python start_live_trading.py &

# Verify system health
sleep 30
curl http://localhost:8000/health

echo "✅ Recovery complete"
```

## 📊 Performance Optimization

### System Optimization

**Memory Management:**
```python
# Add to main.py startup
import gc
gc.set_threshold(700, 10, 10)  # Optimize garbage collection
```

**Database Optimization:**
```sql
-- Add to database initialization
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = memory;
```

**Network Optimization:**
```python
# MT5 connection optimization
MT5_CONNECTION_TIMEOUT = 10
MT5_RETRY_ATTEMPTS = 3
MT5_PING_INTERVAL = 60
```

### Scaling Considerations

**Horizontal Scaling:**
- Deploy multiple instances with load balancer
- Use Redis for shared state management
- Implement distributed signal processing

**Vertical Scaling:**
- Increase server resources (CPU, RAM)
- Optimize database queries and indexing
- Enable GPU acceleration for AI models

## 🔐 Security Hardening

### Access Control

**API Security:**
```python
# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/status")
@limiter.limit("10/minute")
async def get_status(request: Request):
    # ... existing code
```

**Environment Security:**
```bash
# Set secure file permissions
chmod 600 .env
chmod 700 logs/
chmod 755 db/

# Enable firewall
ufw enable
ufw allow 22    # SSH
ufw allow 3000  # Frontend
ufw allow 8000  # Backend API
```

**Database Security:**
```python
# Enable database encryption
DATABASE_URL=file:./db/custom.db?_pragma=key=your_encryption_key
```

## 📞 Support and Maintenance

### Maintenance Schedule

**Daily:**
- Monitor system performance and P&L
- Check log files for errors
- Verify API connections

**Weekly:**
- Review trading performance
- Update AI model training data
- Backup database and configurations

**Monthly:**
- Update dependencies and security patches
- Performance optimization review
- Disaster recovery testing

### Troubleshooting

**Common Issues:**

1. **MT5 Connection Failed**
   - Verify credentials and server
   - Check MT5 terminal connection
   - Restart MT5 terminal if needed

2. **Gemini API Errors**
   - Check API key validity
   - Verify quota limits
   - Monitor rate limiting

3. **High Memory Usage**
   - Restart trading system
   - Check for memory leaks in logs
   - Optimize model loading

4. **Database Lock Errors**
   - Check for concurrent access
   - Restart system if needed
   - Consider database optimization

### Support Resources

- **System Logs**: `./logs/` directory
- **Configuration Files**: `./configs/` directory  
- **Database**: `./db/custom.db`
- **Environment**: `.env` file

---

**⚠️ IMPORTANT**: This is a live trading system that executes real trades with real money. Always test thoroughly in a demo environment before deploying to production. Monitor the system continuously and have emergency procedures ready.
