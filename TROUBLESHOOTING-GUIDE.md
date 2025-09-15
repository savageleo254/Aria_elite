# ARIA ELITE Troubleshooting Guide

**System Diagnostics and Problem Resolution**

This guide provides comprehensive troubleshooting procedures for common issues encountered with the ARIA ELITE trading system.

## 🚨 Emergency Response

### System Not Responding

**Immediate Actions:**
```bash
# 1. Check if processes are running
ps aux | grep -E "(python.*start_live|uvicorn|node)" | grep -v grep

# 2. Check system resources
top -bn1 | head -20
free -h
df -h

# 3. Emergency stop if needed
pkill -f "start_live_trading.py"
pkill -f "uvicorn"

# 4. Check for errors
tail -50 logs/aria_elite_errors.log
```

**Recovery Steps:**
1. Identify the root cause from logs
2. Address the underlying issue
3. Restart system components
4. Verify full functionality
5. Monitor for stability

### Trading System Frozen

**Diagnostic Commands:**
```bash
# Check MT5 connection
python -c "
import MetaTrader5 as mt5
if mt5.initialize():
    print('MT5 Connected')
    print('Account:', mt5.account_info())
    mt5.shutdown()
else:
    print('MT5 Connection Failed')
"

# Check API responsiveness
curl -m 10 http://localhost:8000/health
curl -m 10 -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/status

# Check database locks
sqlite3 db/custom.db "PRAGMA journal_mode;"
fuser db/custom.db 2>/dev/null
```

## 🔌 Connection Issues

### MetaTrader 5 Connection Problems

**Symptoms:**
- "MT5 connection failed" errors
- No trade execution
- Position updates not working

**Diagnostic Steps:**
```bash
# 1. Verify MT5 terminal is running
pgrep -f "terminal64.exe" || pgrep -f "MetaTrader"

# 2. Test direct MT5 connection
python -c "
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()
login = int(os.getenv('MT5_LOGIN'))
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')

if mt5.initialize(login=login, password=password, server=server):
    print('✅ MT5 Connection Successful')
    account = mt5.account_info()
    print(f'Account: {account.login}')
    print(f'Balance: {account.balance}')
    print(f'Server: {account.server}')
else:
    print('❌ MT5 Connection Failed')
    print('Error:', mt5.last_error())
mt5.shutdown()
"
```

**Solutions:**
1. **Invalid Credentials:**
   ```bash
   # Verify environment variables
   echo "Login: $MT5_LOGIN"
   echo "Server: $MT5_SERVER"
   # Password should not be echoed for security
   
   # Test with MT5 terminal directly
   ```

2. **MT5 Terminal Not Running:**
   ```bash
   # Windows
   tasklist | findstr "terminal64"
   
   # Linux (using Wine)
   ps aux | grep terminal64
   
   # Start MT5 terminal
   # Windows: Start MetaTrader 5 application
   # Linux: wine ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal64.exe
   ```

3. **Network/Firewall Issues:**
   ```bash
   # Test server connectivity
   ping your-mt5-server.com
   
   # Check port access (usually 443 for HTTPS)
   telnet your-mt5-server.com 443
   
   # Windows firewall (run as Administrator)
   netsh advfirewall firewall show rule name="MetaTrader 5"
   ```

### Gemini API Connection Issues

**Symptoms:**
- "Gemini connection failed" errors
- No AI analysis in signals
- API timeout errors

**Diagnostic Steps:**
```bash
# 1. Test API key validity
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models" | jq '.models | length'

# 2. Check quota and billing
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models/gemini-pro" | jq '.usage'

# 3. Test simple request
python -c "
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content('Test connection')
    print('✅ Gemini API Working')
    print('Response:', response.text[:100])
except Exception as e:
    print('❌ Gemini API Failed')
    print('Error:', str(e))
"
```

**Solutions:**
1. **Invalid API Key:**
   ```bash
   # Get new API key from Google AI Studio
   # https://makersuite.google.com/app/apikey
   
   # Update environment variable
   nano .env
   # Set: GEMINI_API_KEY=your_new_api_key
   ```

2. **Quota Exceeded:**
   - Check Google Cloud Console for quota limits
   - Upgrade billing plan if necessary
   - Implement request rate limiting

3. **Network Issues:**
   ```bash
   # Test connectivity
   ping generativelanguage.googleapis.com
   
   # Check DNS resolution
   nslookup generativelanguage.googleapis.com
   ```

### Database Connection Issues

**Symptoms:**
- "Database locked" errors
- Slow query performance
- Data inconsistencies

**Diagnostic Steps:**
```bash
# 1. Check database file
ls -la db/custom.db
file db/custom.db

# 2. Test database integrity
sqlite3 db/custom.db "PRAGMA integrity_check;"

# 3. Check for locks
sqlite3 db/custom.db "PRAGMA busy_timeout=30000; SELECT 1;"

# 4. Examine WAL files
ls -la db/custom.db-*
```

**Solutions:**
1. **Database Corruption:**
   ```bash
   # Backup current database
   cp db/custom.db db/custom.db.corrupt
   
   # Try to repair
   sqlite3 db/custom.db ".recover" > db/recovered.sql
   sqlite3 db/custom_new.db < db/recovered.sql
   
   # If successful, replace database
   mv db/custom.db db/custom.db.old
   mv db/custom_new.db db/custom.db
   ```

2. **Lock Issues:**
   ```bash
   # Kill processes using the database
   fuser -k db/custom.db
   
   # Reset WAL mode
   sqlite3 db/custom.db "PRAGMA journal_mode=DELETE; PRAGMA journal_mode=WAL;"
   ```

3. **Performance Issues:**
   ```sql
   -- Run in SQLite
   PRAGMA optimize;
   PRAGMA cache_size = 10000;
   ANALYZE;
   ```

## 🤖 AI Model Issues

### Model Loading Failures

**Symptoms:**
- "Model not found" errors
- Poor prediction accuracy
- Model training failures

**Diagnostic Steps:**
```bash
# 1. Check model files
ls -la backend/models/

# 2. Test model loading
python -c "
from backend.models.ai_models import AIModelManager
import asyncio

async def test_models():
    manager = AIModelManager()
    try:
        await manager.initialize()
        status = await manager.get_model_status()
        print('Model Status:', status)
    except Exception as e:
        print('Error:', str(e))

asyncio.run(test_models())
"

# 3. Check model dependencies
pip show lightgbm tensorflow scikit-learn
```

**Solutions:**
1. **Missing Model Files:**
   ```bash
   # Retrain models
   python backend/scripts/generate_ai_models.py
   
   # Or download pre-trained models (if available)
   # wget https://your-model-repository.com/models.tar.gz
   # tar -xzf models.tar.gz -C backend/models/
   ```

2. **Dependency Issues:**
   ```bash
   # Reinstall ML dependencies
   pip install --upgrade --force-reinstall lightgbm tensorflow scikit-learn
   
   # Check for conflicts
   pip check
   ```

3. **Memory Issues:**
   ```bash
   # Monitor memory during model loading
   free -h
   
   # Increase available memory or reduce model complexity
   export PYTHONHASHSEED=0
   ulimit -v 4194304  # Limit virtual memory to 4GB
   ```

### Poor Trading Performance

**Symptoms:**
- Low win rate (<40%)
- High drawdown
- Inconsistent signals

**Analysis Steps:**
```bash
# 1. Get detailed performance metrics
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  "http://localhost:8000/analytics/performance?period=1w" | jq

# 2. Analyze strategy performance
python -c "
from backend.core.signal_manager import SignalManager
import asyncio

async def analyze_strategies():
    sm = SignalManager()
    await sm.initialize()
    
    strategies = ['smc_strategy', 'ai_strategy', 'news_sentiment']
    for strategy in strategies:
        try:
            perf = await sm.get_strategy_performance(strategy, '1w')
            print(f'{strategy}: {perf.get(\"win_rate\", 0)*100:.1f}% win rate')
        except Exception as e:
            print(f'{strategy}: Error - {e}')

asyncio.run(analyze_strategies())
"

# 3. Check model accuracy
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/models/status | jq '.model_status'
```

**Solutions:**
1. **Retrain Models:**
   ```bash
   # Trigger model retraining with fresh data
   curl -X POST -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
     http://localhost:8000/models/retrain
   
   # Monitor training progress
   tail -f logs/aria_elite.log | grep -E "(training|accuracy)"
   ```

2. **Adjust Strategy Parameters:**
   ```bash
   # Edit strategy configuration
   nano configs/strategy_config.json
   
   # Increase confidence thresholds
   # Reduce position sizes
   # Adjust risk parameters
   ```

3. **Market Condition Analysis:**
   ```bash
   # Check current market volatility
   curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
     "http://localhost:8000/microstructure/analysis?symbol=EURUSD"
   
   # Review news sentiment impact
   curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
     http://localhost:8000/news/sentiment
   ```

## 🔧 System Performance Issues

### High Memory Usage

**Symptoms:**
- System becoming unresponsive
- Out of memory errors
- Slow performance

**Diagnostic Commands:**
```bash
# 1. Check memory usage by process
ps aux --sort=-%mem | head -20

# 2. Check system memory
free -h
cat /proc/meminfo | grep -E "(MemTotal|MemAvailable|MemFree)"

# 3. Monitor memory over time
top -p $(pgrep -f "start_live_trading.py") -d 5

# 4. Check for memory leaks
python -c "
import psutil
import time

process = None
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    if 'start_live_trading.py' in ' '.join(p.info['cmdline'] or []):
        process = p
        break

if process:
    for i in range(10):
        memory = process.memory_info()
        print(f'Memory usage: {memory.rss / 1024 / 1024:.1f} MB')
        time.sleep(30)
"
```

**Solutions:**
1. **Optimize Python Memory:**
   ```bash
   # Set garbage collection thresholds
   export PYTHONOPTIMIZE=1
   export PYTHONDONTWRITEBYTECODE=1
   
   # Restart with memory optimization
   python -c "
   import gc
   gc.set_threshold(700, 10, 10)
   exec(open('start_live_trading.py').read())
   "
   ```

2. **Database Optimization:**
   ```bash
   # Optimize database memory usage
   sqlite3 db/custom.db "
   PRAGMA cache_size = 5000;
   PRAGMA temp_store = memory;
   PRAGMA mmap_size = 134217728;
   "
   ```

3. **Model Memory Management:**
   ```python
   # Add to model loading code
   import tensorflow as tf
   
   # Limit GPU memory growth
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

### High CPU Usage

**Symptoms:**
- System lag
- Slow response times
- High load average

**Diagnostic Steps:**
```bash
# 1. Check CPU usage
top -bn1 | head -20
htop -p $(pgrep -f "start_live_trading.py")

# 2. Profile Python code
python -m cProfile -o profile_output.prof start_live_trading.py &
sleep 300  # Let it run for 5 minutes
kill $!

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"

# 3. Check I/O wait
iostat -x 1 5
```

**Solutions:**
1. **Optimize Critical Paths:**
   ```python
   # Use connection pooling for database
   # Implement caching for frequent queries
   # Optimize model inference batching
   ```

2. **Adjust Process Priorities:**
   ```bash
   # Lower priority for non-critical processes
   renice +5 $(pgrep -f "uvicorn")
   
   # Higher priority for trading engine
   renice -5 $(pgrep -f "start_live_trading.py")
   ```

3. **System Tuning:**
   ```bash
   # Increase file descriptor limits
   ulimit -n 65536
   
   # Optimize scheduler
   echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
   ```

## 🌐 Network and API Issues

### API Rate Limiting

**Symptoms:**
- "Too Many Requests" errors
- 429 HTTP status codes
- Delayed responses

**Solutions:**
```python
# Implement exponential backoff
import time
import random

def api_request_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

### Discord Bot Issues

**Symptoms:**
- Bot not responding to commands
- Missing notifications
- Connection timeouts

**Diagnostic Steps:**
```bash
# 1. Check bot status
python -c "
import discord
import asyncio
import os

async def test_bot():
    client = discord.Client()
    
    @client.event
    async def on_ready():
        print(f'Bot connected as {client.user}')
        await client.close()
    
    try:
        await client.start(os.getenv('DISCORD_BOT_TOKEN'))
    except Exception as e:
        print(f'Bot connection failed: {e}')

asyncio.run(test_bot())
"

# 2. Test webhook
curl -X POST -H "Content-Type: application/json" \
  -d '{"content": "Test message"}' \
  "$DISCORD_WEBHOOK_URL"
```

**Solutions:**
1. **Token Issues:**
   - Verify bot token in Discord Developer Portal
   - Regenerate token if compromised
   - Update environment variable

2. **Permission Issues:**
   - Check bot permissions in Discord server
   - Ensure bot has "Send Messages" permission
   - Re-invite bot with correct permissions

## 📊 Data Quality Issues

### Missing Market Data

**Symptoms:**
- Incomplete price history
- Signal generation failures
- Data gaps in analysis

**Diagnostic Steps:**
```bash
# 1. Check data completeness
sqlite3 db/custom.db "
SELECT symbol, COUNT(*) as records, 
       MIN(timestamp) as oldest,
       MAX(timestamp) as newest
FROM MarketData 
GROUP BY symbol;
"

# 2. Test data source
python -c "
import yfinance as yf
import pandas as pd

try:
    data = yf.download('EURUSD=X', period='1d', interval='1h')
    print(f'Downloaded {len(data)} records')
    print('Latest timestamp:', data.index[-1])
except Exception as e:
    print(f'Data download failed: {e}')
"
```

**Solutions:**
1. **Data Source Issues:**
   ```python
   # Implement multiple data sources
   providers = ['yahoo', 'alpha_vantage', 'polygon']
   for provider in providers:
       try:
           data = fetch_data(provider, symbol, timeframe)
           if data is not None:
               break
       except Exception as e:
           continue
   ```

2. **Data Backfill:**
   ```bash
   # Backfill missing data
   python -c "
   from backend.core.data_engine import DataEngine
   import asyncio
   
   async def backfill_data():
       engine = DataEngine()
       await engine.initialize()
       
       symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
       for symbol in symbols:
           await engine.backfill_historical_data(symbol, days=30)
           print(f'Backfilled {symbol}')
   
   asyncio.run(backfill_data())
   "
   ```

## 🔄 System Recovery Procedures

### Complete System Recovery

**Recovery Checklist:**
```bash
#!/bin/bash
# system_recovery.sh

echo "=== ARIA ELITE System Recovery ==="
date

# 1. Stop all processes
echo "Stopping all processes..."
pkill -f "start_live_trading.py"
pkill -f "uvicorn"
sleep 10

# 2. Check system resources
echo "Checking system resources..."
free -h
df -h
uptime

# 3. Verify database integrity
echo "Checking database..."
sqlite3 db/custom.db "PRAGMA integrity_check;" | head -1

# 4. Test external connections
echo "Testing connections..."
curl -m 10 https://finance.yahoo.com > /dev/null && echo "✅ Yahoo Finance OK" || echo "❌ Yahoo Finance Failed"
curl -m 10 -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models" > /dev/null && echo "✅ Gemini API OK" || echo "❌ Gemini API Failed"

# 5. Restart system
echo "Restarting system..."
python start_live_trading.py &
sleep 30

# 6. Verify recovery
echo "Verifying recovery..."
curl -m 10 http://localhost:8000/health && echo "✅ System recovered" || echo "❌ Recovery failed"

echo "Recovery completed: $(date)"
```

### Backup and Restore

**Create Emergency Backup:**
```bash
#!/bin/bash
# emergency_backup.sh

timestamp=$(date +%Y%m%d_%H%M%S)
backup_dir="emergency_backup_$timestamp"

mkdir -p "$backup_dir"

# Backup critical files
cp -r configs/ "$backup_dir/"
cp -r db/ "$backup_dir/"
cp -r logs/ "$backup_dir/"
cp .env "$backup_dir/"
cp -r backend/models/ "$backup_dir/"

# Create archive
tar -czf "$backup_dir.tar.gz" "$backup_dir"
rm -rf "$backup_dir"

echo "Emergency backup created: $backup_dir.tar.gz"
```

**Restore from Backup:**
```bash
#!/bin/bash
# restore_backup.sh

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

backup_file="$1"

# Stop system
pkill -f "start_live_trading.py"

# Extract backup
tar -xzf "$backup_file"
backup_dir="${backup_file%.tar.gz}"

# Restore files
cp -r "$backup_dir/configs/" .
cp -r "$backup_dir/db/" .
cp -r "$backup_dir/backend/models/" backend/
cp "$backup_dir/.env" .

# Set permissions
chmod 600 .env
chmod -R 755 configs/

# Restart system
python start_live_trading.py &

echo "System restored from backup: $backup_file"
```

## 📞 Getting Help

### Log Analysis

**Key Log Files:**
- `logs/aria_elite.log` - Main system log
- `logs/aria_elite_errors.log` - Error-only log
- `logs/trading_activity.log` - Trading operations log
- `logs/backend.core.*.log` - Component-specific logs

**Useful Log Analysis Commands:**
```bash
# Find critical errors in last 24 hours
find logs/ -name "*.log" -mtime -1 -exec grep -l "CRITICAL\|FATAL" {} \;

# Count error types
grep "ERROR" logs/aria_elite.log | awk '{print $4}' | sort | uniq -c | sort -nr

# Track specific symbol trading
grep "EURUSD" logs/trading_activity.log | tail -20

# Monitor live logs
tail -f logs/aria_elite.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

### Support Information

**Before Contacting Support:**
1. Gather system information:
   ```bash
   # System specs
   uname -a
   python --version
   node --version
   
   # Error logs
   tail -100 logs/aria_elite_errors.log
   
   # System status
   curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
     http://localhost:8000/status
   ```

2. Document the issue:
   - What were you trying to do?
   - What happened instead?
   - When did it start?
   - What changes were made recently?

3. Include relevant configuration (without secrets):
   ```bash
   # Safe config info
   cat configs/strategy_config.json | jq '.strategies | keys'
   cat configs/execution_config.json | jq '.risk_management'
   ```

---

**🆘 Emergency Support**: For critical issues affecting live trading, immediately activate the kill switch and contact support with full system logs and error details.
