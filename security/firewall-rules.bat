@echo off
REM ARIA ELITE Production Firewall Configuration for Windows
REM This script configures Windows Firewall for production deployment

echo 🔒 Configuring Windows Firewall for ARIA ELITE Production...

REM Note: This requires administrator privileges
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script requires administrator privileges.
    echo Please run as administrator.
    pause
    exit /b 1
)

REM Allow HTTP (port 80)
netsh advfirewall firewall add rule name="ARIA ELITE HTTP" dir=in action=allow protocol=TCP localport=80

REM Allow HTTPS (port 443)
netsh advfirewall firewall add rule name="ARIA ELITE HTTPS" dir=in action=allow protocol=TCP localport=443

REM Allow SSH (port 22) - adjust if using different port
netsh advfirewall firewall add rule name="ARIA ELITE SSH" dir=in action=allow protocol=TCP localport=22

REM Optional: Allow monitoring ports (uncomment if needed)
REM netsh advfirewall firewall add rule name="ARIA ELITE Grafana" dir=in action=allow protocol=TCP localport=3001
REM netsh advfirewall firewall add rule name="ARIA ELITE Prometheus" dir=in action=allow protocol=TCP localport=9090

REM Block unnecessary ports
netsh advfirewall firewall add rule name="Block RDP" dir=in action=block protocol=TCP localport=3389
netsh advfirewall firewall add rule name="Block Telnet" dir=in action=block protocol=TCP localport=23
netsh advfirewall firewall add rule name="Block FTP" dir=in action=block protocol=TCP localport=21

echo [INFO] Firewall rules configured successfully!
echo [INFO] Allowed ports:
echo [INFO]   - HTTP (80)
echo [INFO]   - HTTPS (443)
echo [INFO]   - SSH (22)
echo [INFO] Blocked ports:
echo [INFO]   - RDP (3389)
echo [INFO]   - Telnet (23)
echo [INFO]   - FTP (21)

echo [WARNING] Make sure you have remote access configured before applying these rules!

pause
