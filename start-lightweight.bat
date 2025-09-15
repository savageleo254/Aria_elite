@echo off
title ARIA ELITE - Lightweight Deployment

echo ===============================================
echo ARIA ELITE - ThinkPad T470 Optimized Startup
echo ===============================================
echo Hardware: i5-7th Gen, 8GB RAM, 256GB SSD
echo ===============================================

echo Checking system resources...
wmic computersystem get TotalPhysicalMemory /value
wmic logicaldisk get size,freespace /value

echo.
echo Starting ARIA ELITE in lightweight mode...
echo - Memory limit: 2GB for backend
echo - Redis cache: 512MB
echo - Tick buffer: 2K (reduced from 10K)
echo - Analysis window: 2 minutes
echo - Worker processes: 4
echo.

REM Set environment variables for lightweight mode
set MEMORY_EFFICIENT_MODE=true
set REDIS_MAX_MEMORY=512mb
set TICK_BUFFER_SIZE=2000
set ANALYSIS_WINDOW=120
set MAX_WORKER_THREADS=6

REM Start Redis with memory optimization
echo Starting Redis with memory optimization...
start /B redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

REM Wait for Redis to start
timeout /t 5 /nobreak >nul

REM Start backend with resource monitoring
echo Starting ARIA backend with hardware optimization...
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2 --limit-concurrency 50

pause
