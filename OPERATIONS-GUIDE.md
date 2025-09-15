# ARIA ELITE Operations Guide

**Daily Operations and Trading System Management**

This guide covers the day-to-day operations, monitoring, and management of the ARIA ELITE trading system.

## 🚀 System Startup

### Pre-Flight Checklist

Before starting live trading operations, verify:

```bash
# 1. Check system requirements
python --version  # Should be 3.9+
node --version    # Should be 18+

# 2. Verify environment variables
echo $GEMINI_API_KEY
echo $MT5_LOGIN
echo $SUPERVISOR_API_TOKEN

# 3. Check database status
ls -la db/custom.db
npm run db:generate

# 4. Verify MT5 connection
# Open MT5 terminal and ensure it's connected

# 5. Test Gemini API
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models"
```

### System Startup Sequence

**Method 1: Complete System Startup**
```bash
# Start complete trading system
python start_live_trading.py
```

**Method 2: Component-by-Component Startup**
```bash
# Terminal 1: Start backend API
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend dashboard
npm run start

# Terminal 3: Start trading system
python start_live_trading.py
```

**Method 3: Docker Startup**
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

### Startup Verification

**System Health Check:**
```bash
# API health
curl http://localhost:8000/health

# System status
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/status

# Frontend access
curl http://localhost:3000
```

**Expected Startup Log Sequence:**
```
🚀 Initializing ARIA ELITE Trading System...
📊 Initializing data engine...
🤖 Initializing AI model manager...
📈 Initializing SMC module...
🧠 Initializing Gemini workflow agent...
📡 Initializing signal manager...
⚡ Initializing execution engine...
🔍 Initializing Gemini audit agent...
💬 Initializing Discord notifications...
✅ All components initialized successfully!
🔍 Running initial system audit...
🎯 Starting live trading operations...
```

## 📊 Daily Operations

### Morning Routine (Market Open)

**1. System Status Check (5 minutes)**
```bash
# Check overnight performance
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  "http://localhost:8000/analytics/performance?period=1d"

# Verify active positions
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/trades/active

# Check system logs for errors
tail -50 logs/aria_elite.log | grep ERROR
```

**2. Market Conditions Assessment (10 minutes)**
```bash
# Get current market sentiment
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/news/sentiment

# Check model status
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/models/status

# Review microstructure analysis
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  "http://localhost:8000/microstructure/analysis?symbol=EURUSD"
```

**3. System Configuration Review (5 minutes)**
- Verify trading sessions are correctly configured
- Check risk limits and position sizes
- Review any overnight alerts or notifications
- Confirm Discord notifications are working

### Intraday Monitoring

**Continuous Monitoring (Every 30 minutes)**

**Performance Dashboard:**
- Navigate to: `http://localhost:3000`
- Monitor real-time P&L
- Check active positions and risk exposure
- Review recent signals and execution quality

**System Metrics:**
```bash
# System resource usage
ps aux | grep -E "(python|node)" | head -10

# Memory usage
free -h

# Disk space
df -h

# Log file sizes
du -sh logs/*
```

**Trading Activity:**
```bash
# Recent trades
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  "http://localhost:8000/trades/history?limit=10"

# Active signals
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/status | jq '.active_signals'

# Win rate check
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/status | jq '.win_rate'
```

### Evening Routine (Market Close)

**1. Daily Performance Review (10 minutes)**
```bash
# Generate daily report
python -c "
from backend.core.signal_manager import SignalManager
import asyncio
async def daily_report():
    sm = SignalManager()
    await sm.initialize()
    metrics = await sm.get_performance_metrics('1d')
    print(f'Daily P&L: ${metrics[\"total_pnl\"]:.2f}')
    print(f'Win Rate: {metrics[\"win_rate\"]*100:.1f}%')
    print(f'Total Trades: {metrics[\"total_trades\"]}')
asyncio.run(daily_report())
"
```

**2. System Maintenance (15 minutes)**
```bash
# Rotate logs
find logs/ -name "*.log" -size +100M -exec gzip {} \;

# Clean up old log files (keep 30 days)
find logs/ -name "*.gz" -mtime +30 -delete

# Database optimization
sqlite3 db/custom.db "PRAGMA optimize;"

# Check disk usage
du -sh db/ logs/ backend/models/
```

**3. Backup Critical Data**
```bash
# Backup database
cp db/custom.db db/backup/custom_$(date +%Y%m%d).db

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz configs/

# Backup recent logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## 🎯 Trading Operations

### Signal Generation Management

**Manual Signal Generation:**
```bash
# Generate signal for specific pair
curl -X POST http://localhost:8000/signals/generate \
  -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "1h",
    "strategy": "smc_strategy"
  }'
```

**Strategy Performance Monitoring:**
```bash
# SMC strategy performance
grep "smc_strategy" logs/trading_activity.log | tail -20

# AI strategy performance
grep "ai_strategy" logs/trading_activity.log | tail -20

# News sentiment performance
grep "news_sentiment" logs/trading_activity.log | tail -20
```

### Position Management

**Active Position Monitoring:**
```bash
# Get detailed position information
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/trades/active | jq '.active_trades[]'

# Check position risk exposure
python -c "
import requests
import json
import os

token = os.getenv('SUPERVISOR_API_TOKEN')
response = requests.get(
    'http://localhost:8000/trades/active',
    headers={'Authorization': f'Bearer {token}'}
)
trades = response.json()['active_trades']
total_risk = sum(abs(trade['volume']) for trade in trades)
print(f'Total Position Risk: {total_risk:.2f} lots')
"
```

**Position Adjustment:**
```bash
# Close specific position (if needed)
# Note: Use with caution - this closes actual trades
curl -X POST http://localhost:8000/trades/close \
  -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trade_id": "trade_12345"}'
```

### Risk Management

**Daily Risk Monitoring:**
```python
#!/usr/bin/env python3
# daily_risk_check.py

import asyncio
import os
from backend.core.signal_manager import SignalManager

async def check_risk_metrics():
    sm = SignalManager()
    await sm.initialize()
    
    status = await sm.get_system_status()
    
    # Risk alerts
    if status['daily_pnl'] < -0.03 * 10000:  # 3% daily loss
        print("🚨 ALERT: Daily loss exceeds 3%")
    
    if len(status.get('active_positions', [])) > 5:
        print("🚨 ALERT: Too many active positions")
    
    if status['win_rate'] < 0.4:
        print("⚠️ WARNING: Win rate below 40%")
    
    print(f"Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"Active Positions: {status['active_positions']}")
    print(f"Win Rate: {status['win_rate']*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(check_risk_metrics())
```

## 🔧 System Administration

### Configuration Management

**Update Trading Parameters:**
```bash
# Edit strategy configuration
nano configs/strategy_config.json

# Validate JSON syntax
python -m json.tool configs/strategy_config.json

# Restart system to apply changes
pkill -f "start_live_trading.py"
sleep 5
python start_live_trading.py &
```

**Environment Variable Updates:**
```bash
# Update environment file
nano .env

# Verify changes
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GEMINI_API_KEY:', 'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET')"

# Restart system
systemctl restart aria-elite  # If using systemd
```

### AI Model Management

**Model Status Check:**
```bash
# Check model performance
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/models/status

# List model files
ls -la backend/models/*.pkl backend/models/*.keras backend/models/*.onnx
```

**Model Retraining:**
```bash
# Trigger model retraining (background process)
curl -X POST http://localhost:8000/models/retrain \
  -H "Authorization: Bearer $SUPERVISOR_API_TOKEN"

# Monitor retraining progress
tail -f logs/aria_elite.log | grep -E "(training|model|accuracy)"

# Manual model training
python backend/scripts/retrain_models_mt5_data.py
```

### Database Administration

**Database Maintenance:**
```bash
# Check database size
du -h db/custom.db

# Database integrity check
sqlite3 db/custom.db "PRAGMA integrity_check;"

# Vacuum database (compact)
sqlite3 db/custom.db "VACUUM;"

# Database statistics
sqlite3 db/custom.db "
SELECT name, COUNT(*) as row_count 
FROM sqlite_master sm
JOIN sqlite_temp_master stm ON sm.name = stm.name
WHERE sm.type = 'table'
GROUP BY sm.name;
"
```

**Data Cleanup:**
```bash
# Clean old market data (keep 90 days)
sqlite3 db/custom.db "
DELETE FROM MarketData 
WHERE timestamp < datetime('now', '-90 days');
"

# Clean old trade history (keep 365 days)
sqlite3 db/custom.db "
DELETE FROM Trade 
WHERE createdAt < datetime('now', '-365 days') 
AND status = 'closed';
"

# Clean old logs (keep 30 days)
sqlite3 db/custom.db "
DELETE FROM SystemLog 
WHERE createdAt < datetime('now', '-30 days');
"
```

## 📈 Performance Optimization

### System Performance Monitoring

**Resource Usage Monitoring:**
```bash
#!/bin/bash
# performance_monitor.sh

echo "=== System Performance Report ==="
echo "Timestamp: $(date)"
echo

# CPU usage
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'

# Memory usage
echo "Memory Usage:"
free -h | grep Mem | awk '{print $3 "/" $2}'

# Disk usage
echo "Disk Usage:"
df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}'

# Process information
echo "ARIA ELITE Processes:"
ps aux | grep -E "(python.*start_live|uvicorn|node.*server)" | grep -v grep

# Network connections
echo "Network Connections:"
netstat -an | grep -E ":3000|:8000" | wc -l

# Log file sizes
echo "Log File Sizes:"
du -h logs/*.log 2>/dev/null | sort -hr | head -5
```

**Performance Optimization:**
```bash
# Optimize Python garbage collection
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Node.js memory optimization
export NODE_OPTIONS="--max-old-space-size=2048"

# Set process priorities
renice -10 $(pgrep -f "start_live_trading.py")
renice -5 $(pgrep -f "uvicorn")
```

### Database Performance

**Query Optimization:**
```sql
-- Add indexes for frequently queried tables
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
ON MarketData(symbolId, timestamp);

CREATE INDEX IF NOT EXISTS idx_trades_status_timestamp 
ON Trade(status, createdAt);

CREATE INDEX IF NOT EXISTS idx_signals_status_confidence 
ON Signal(status, confidence);

-- Update table statistics
ANALYZE;
```

## 🚨 Emergency Procedures

### Emergency Stop (Kill Switch)

**Immediate Stop - All Methods:**
```bash
# Method 1: API kill switch
curl -X POST http://localhost:8000/system/kill-switch \
  -H "Authorization: Bearer $SUPERVISOR_API_TOKEN"

# Method 2: Discord command (if configured)
# In Discord: !kill-switch

# Method 3: Direct process termination
pkill -f "start_live_trading.py"
pkill -f "uvicorn"

# Method 4: System service stop
systemctl stop aria-elite
```

**Emergency Checklist:**
1. ✅ Stop all trading operations
2. ✅ Close all open positions
3. ✅ Verify no pending orders
4. ✅ Check account balance
5. ✅ Document the incident
6. ✅ Review logs for cause

### System Recovery

**Recovery Procedure:**
```bash
# 1. Assess the situation
tail -100 logs/aria_elite_errors.log

# 2. Check MT5 connection
python -c "
from backend.core.mt5_bridge import MT5Bridge
import asyncio
async def test():
    bridge = MT5Bridge()
    await bridge.initialize()
    print('MT5 Connected:', bridge.is_connected())
asyncio.run(test())
"

# 3. Verify Gemini API
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models" | jq '.models | length'

# 4. Check database integrity
sqlite3 db/custom.db "PRAGMA integrity_check;"

# 5. Restart system components
python start_live_trading.py

# 6. Verify recovery
curl http://localhost:8000/health
```

### Disaster Recovery

**System Backup Restoration:**
```bash
# Stop all services
systemctl stop aria-elite

# Restore database backup
cp db/backup/custom_$(date +%Y%m%d).db db/custom.db

# Restore configuration
tar -xzf config_backup_$(date +%Y%m%d).tar.gz

# Restart services
systemctl start aria-elite

# Verify restoration
curl -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/status
```

## 📊 Reporting and Analytics

### Daily Performance Report

**Automated Daily Report:**
```python
#!/usr/bin/env python3
# daily_report.py

import asyncio
import json
from datetime import datetime
from backend.core.signal_manager import SignalManager
from backend.utils.discord_notifier import discord_notifier

async def generate_daily_report():
    sm = SignalManager()
    await sm.initialize()
    
    # Get performance metrics
    metrics = await sm.get_performance_metrics('1d')
    
    report = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "summary": {
            "total_pnl": metrics['total_pnl'],
            "total_trades": metrics['total_trades'],
            "win_rate": metrics['win_rate'],
            "profit_factor": metrics.get('profit_factor', 0),
            "max_drawdown": metrics.get('max_drawdown', 0)
        },
        "system_health": await sm.get_system_status()
    }
    
    # Save report
    with open(f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Send to Discord
    if discord_notifier:
        await discord_notifier.send_daily_report(report)
    
    print(f"Daily Report Generated: {report}")

if __name__ == "__main__":
    asyncio.run(generate_daily_report())
```

### Weekly Analysis

**Weekly Performance Review:**
```bash
# Generate weekly report
python -c "
from backend.core.signal_manager import SignalManager
import asyncio
import json

async def weekly_analysis():
    sm = SignalManager()
    await sm.initialize()
    
    metrics = await sm.get_performance_metrics('1w')
    
    print('=== WEEKLY PERFORMANCE SUMMARY ===')
    print(f'Total P&L: ${metrics[\"total_pnl\"]:.2f}')
    print(f'Total Trades: {metrics[\"total_trades\"]}')
    print(f'Win Rate: {metrics[\"win_rate\"]*100:.1f}%')
    print(f'Profit Factor: {metrics.get(\"profit_factor\", 0):.2f}')
    print(f'Max Drawdown: {metrics.get(\"max_drawdown\", 0)*100:.1f}%')
    
    # Strategy breakdown
    strategies = ['smc_strategy', 'ai_strategy', 'news_sentiment']
    for strategy in strategies:
        try:
            strategy_metrics = await sm.get_strategy_performance(strategy, '1w')
            print(f'{strategy}: {strategy_metrics[\"win_rate\"]*100:.1f}% win rate')
        except:
            pass

asyncio.run(weekly_analysis())
"
```

### Monthly Maintenance

**Monthly Tasks Checklist:**
```bash
#!/bin/bash
# monthly_maintenance.sh

echo "=== ARIA ELITE Monthly Maintenance ==="
date

# 1. Model performance review
echo "1. Checking AI model performance..."
curl -s -H "Authorization: Bearer $SUPERVISOR_API_TOKEN" \
  http://localhost:8000/models/status | jq '.model_status'

# 2. Database maintenance
echo "2. Database maintenance..."
sqlite3 db/custom.db "VACUUM; PRAGMA optimize;"

# 3. Log rotation
echo "3. Log file cleanup..."
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;
find logs/ -name "*.gz" -mtime +90 -delete

# 4. Configuration backup
echo "4. Backing up configurations..."
tar -czf "backups/monthly_backup_$(date +%Y%m).tar.gz" configs/ db/ logs/

# 5. System updates check
echo "5. Checking for updates..."
pip list --outdated | head -10
npm outdated

# 6. Performance summary
echo "6. Performance summary..."
python daily_report.py

echo "Monthly maintenance completed: $(date)"
```

---

**📞 Operations Support**: For operational questions, system issues, or emergency support, refer to the troubleshooting guide or contact the development team immediately.
