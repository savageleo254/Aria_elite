@echo off
REM ARIA ELITE Production Stop Script for Windows
setlocal enabledelayedexpansion

echo 🛑 Stopping ARIA ELITE Production Services...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running.
    pause
    exit /b 1
)

REM Stop services
echo [INFO] Stopping production services...
docker-compose -f docker-compose.prod.yml down

REM Show final status
echo [INFO] Service Status:
docker-compose -f docker-compose.prod.yml ps

echo [INFO] 🎉 Production services stopped successfully!

pause
