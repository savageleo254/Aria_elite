#!/bin/bash

# ARIA ELITE Production Firewall Configuration
# This script configures UFW (Uncomplicated Firewall) for production deployment

echo "🔒 Configuring firewall for ARIA ELITE Production..."

# Reset firewall to default
ufw --force reset

# Set default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (adjust port if needed)
ufw allow 22/tcp comment 'SSH'

# Allow HTTP and HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Allow monitoring ports (optional - only if needed externally)
# ufw allow 3001/tcp comment 'Grafana'
# ufw allow 9090/tcp comment 'Prometheus'

# Allow specific IP ranges for admin access (replace with your IPs)
# ufw allow from 192.168.1.0/24 to any port 22 comment 'Admin SSH'
# ufw allow from 10.0.0.0/8 to any port 3001 comment 'Admin Grafana'

# Enable firewall
ufw --force enable

# Show status
echo "Firewall status:"
ufw status verbose

echo "✅ Firewall configured successfully!"
echo "⚠️  Make sure you have SSH access before enabling firewall on remote servers!"
