@echo off
REM ARIA ELITE Production Startup Script for Windows
setlocal enabledelayedexpansion

echo 🚀 Starting ARIA ELITE Production Services...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo [ERROR] No .env file found. Please create one from production.env.example
    pause
    exit /b 1
)

REM Start services
echo [INFO] Starting production services...
docker-compose -f docker-compose.prod.yml up -d

REM Wait for services to start
echo [INFO] Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Show service status
echo [INFO] Service Status:
docker-compose -f docker-compose.prod.yml ps

echo [INFO] 🎉 Production services started!
echo [INFO] Services are available at:
echo [INFO]   - Frontend: https://localhost
echo [INFO]   - Backend API: https://localhost/api
echo [INFO]   - Supervisor: https://localhost/supervisor
echo [INFO]   - Grafana: http://localhost:3001
echo [INFO]   - Prometheus: http://localhost:9090

echo [INFO] To view logs: docker-compose -f docker-compose.prod.yml logs -f
echo [INFO] To stop services: docker-compose -f docker-compose.prod.yml down

pause
