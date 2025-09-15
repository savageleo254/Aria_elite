# ARIA ELITE Configuration Guide

**Complete Configuration Reference for Trading System**

This guide covers all configuration aspects of the ARIA ELITE trading system, from environment setup to advanced trading parameters.

## 🔧 Environment Configuration

### Core Environment Variables

Create `.env` file in the project root:

```bash
# =============================================================================
# ARIA ELITE PRODUCTION CONFIGURATION
# =============================================================================

# System Environment
NODE_ENV=production
DATABASE_URL=file:./db/custom.db

# =============================================================================
# API KEYS AND AUTHENTICATION
# =============================================================================

# Google Gemini AI API
GEMINI_API_KEY=your_gemini_api_key_here

# System Security Tokens
SUPERVISOR_API_TOKEN=your_secure_supervisor_token
JWT_SECRET=your_secure_jwt_secret
ENCRYPTION_KEY=your_secure_encryption_key

# =============================================================================
# TRADING PLATFORM INTEGRATION
# =============================================================================

# MetaTrader 5 Connection
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# =============================================================================
# DISCORD INTEGRATION (OPTIONAL)
# =============================================================================

# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_WEBHOOK_URL=your_discord_webhook_url
DISCORD_ALLOWED_USERS=user_id_1,user_id_2,user_id_3

# =============================================================================
# SYSTEM PERFORMANCE
# =============================================================================

# Logging Configuration
LOG_LEVEL=info
LOG_FILE_PATH=./logs
LOG_ROTATION=daily
LOG_RETENTION_DAYS=30

# Performance Settings
MAX_WORKERS=4
WORKER_TIMEOUT=30
MEMORY_LIMIT=2048

# =============================================================================
# BACKEND API CONFIGURATION
# =============================================================================

# Backend API Settings
BACKEND_API_BASE=http://127.0.0.1:8000
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend Settings
NEXT_PUBLIC_API_URL=http://localhost:8000
REACT_APP_API_URL=http://localhost:8000

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# CORS Configuration
CORS_ORIGIN=http://localhost:3000,http://127.0.0.1:3000

# Session Management
SESSION_SECRET=your_secure_session_secret
SESSION_TIMEOUT=3600

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_SIGNAL_GENERATION=10
RATE_LIMIT_TRADE_EXECUTION=30
```

### Environment Variable Descriptions

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Google Gemini AI API key | ✅ Yes | - |
| `MT5_LOGIN` | MetaTrader 5 account login | ✅ Yes | - |
| `MT5_PASSWORD` | MetaTrader 5 account password | ✅ Yes | - |
| `MT5_SERVER` | MetaTrader 5 server address | ✅ Yes | - |
| `SUPERVISOR_API_TOKEN` | API authentication token | ✅ Yes | - |
| `DISCORD_BOT_TOKEN` | Discord bot token | ❌ No | - |
| `LOG_LEVEL` | Logging verbosity | ❌ No | `info` |
| `MAX_WORKERS` | Number of worker processes | ❌ No | `4` |

## 📊 Trading Configuration

### Strategy Configuration (`configs/strategy_config.json`)

```json
{
  "version": "1.2.0",
  "last_updated": "2024-01-01T00:00:00.000Z",
  
  "global_settings": {
    "enabled": true,
    "max_concurrent_signals": 10,
    "signal_expiry_minutes": 60,
    "min_confidence_threshold": 0.65
  },
  
  "strategies": {
    "smc_strategy": {
      "enabled": true,
      "weight": 0.4,
      "parameters": {
        "confidence_threshold": 0.75,
        "structure_break_confirmation": true,
        "liquidity_grab_detection": true,
        "order_block_validation": true,
        "fair_value_gap_analysis": true,
        "timeframe_alignment": ["1h", "4h"],
        "risk_reward_ratio": 2.5
      },
      "risk_management": {
        "max_risk_per_trade": 0.02,
        "max_concurrent_trades": 3,
        "stop_loss_atr_multiplier": 2.0,
        "take_profit_atr_multiplier": 3.0
      }
    },
    
    "ai_strategy": {
      "enabled": true,
      "weight": 0.4,
      "parameters": {
        "model_ensemble": {
          "lightgbm_weight": 0.4,
          "cnn_weight": 0.3,
          "mobilenet_weight": 0.3
        },
        "prediction_threshold": 0.7,
        "feature_importance_threshold": 0.15,
        "model_agreement_required": 2,
        "lookback_periods": 100
      },
      "risk_management": {
        "max_risk_per_trade": 0.015,
        "max_concurrent_trades": 2,
        "confidence_scaling": true,
        "dynamic_position_sizing": true
      }
    },
    
    "news_sentiment": {
      "enabled": true,
      "weight": 0.2,
      "parameters": {
        "sentiment_threshold": 0.6,
        "impact_filter": "medium",
        "source_reliability_weight": true,
        "sentiment_decay_hours": 4,
        "correlation_analysis": true
      },
      "risk_management": {
        "max_risk_per_trade": 0.01,
        "max_concurrent_trades": 1,
        "news_impact_scaling": true
      }
    }
  },
  
  "symbols": {
    "EURUSD": {
      "enabled": true,
      "spread_filter": 0.0002,
      "volatility_filter": 0.001,
      "trading_sessions": ["london", "new_york"]
    },
    "GBPUSD": {
      "enabled": true,
      "spread_filter": 0.0003,
      "volatility_filter": 0.0012,
      "trading_sessions": ["london", "new_york"]
    },
    "USDJPY": {
      "enabled": true,
      "spread_filter": 0.002,
      "volatility_filter": 0.01,
      "trading_sessions": ["tokyo", "london"]
    },
    "AUDUSD": {
      "enabled": false,
      "spread_filter": 0.0003,
      "volatility_filter": 0.0015,
      "trading_sessions": ["sydney", "tokyo"]
    }
  },
  
  "timeframes": {
    "1m": { "enabled": false, "weight": 0.1 },
    "5m": { "enabled": false, "weight": 0.15 },
    "15m": { "enabled": true, "weight": 0.2 },
    "1h": { "enabled": true, "weight": 0.3 },
    "4h": { "enabled": true, "weight": 0.25 },
    "1d": { "enabled": false, "weight": 0.0 }
  }
}
```

### Execution Configuration (`configs/execution_config.json`)

```json
{
  "version": "1.1.0",
  "last_updated": "2024-01-01T00:00:00.000Z",
  
  "risk_management": {
    "global_limits": {
      "max_daily_loss_percentage": 0.05,
      "max_monthly_loss_percentage": 0.15,
      "max_drawdown_percentage": 0.20,
      "max_concurrent_positions": 5,
      "max_correlation_exposure": 0.7
    },
    
    "position_sizing": {
      "method": "atr_based",
      "default_risk_percentage": 0.02,
      "min_position_size": 0.01,
      "max_position_size": 0.1,
      "confidence_scaling": true,
      "volatility_adjustment": true
    },
    
    "stop_loss": {
      "method": "atr_based",
      "atr_multiplier": 2.0,
      "min_stop_loss_pips": 10,
      "max_stop_loss_pips": 100,
      "trailing_stop": false
    },
    
    "take_profit": {
      "method": "atr_based",
      "atr_multiplier": 3.0,
      "partial_profit_taking": true,
      "profit_levels": [0.5, 0.3, 0.2]
    }
  },
  
  "execution_settings": {
    "order_execution": {
      "default_order_type": "market",
      "slippage_tolerance": 0.0005,
      "max_execution_time_seconds": 30,
      "retry_attempts": 3,
      "retry_delay_seconds": 5
    },
    
    "market_conditions": {
      "spread_filter_enabled": true,
      "max_spread_multiplier": 2.0,
      "volatility_filter_enabled": true,
      "low_liquidity_detection": true,
      "news_event_filtering": true
    },
    
    "timing": {
      "trading_sessions": {
        "sydney": { "start": "22:00", "end": "07:00", "enabled": false },
        "tokyo": { "start": "00:00", "end": "09:00", "enabled": true },
        "london": { "start": "08:00", "end": "17:00", "enabled": true },
        "new_york": { "start": "13:00", "end": "22:00", "enabled": true }
      },
      "avoid_news_minutes_before": 15,
      "avoid_news_minutes_after": 15,
      "weekend_trading": false
    }
  },
  
  "monitoring": {
    "position_monitoring": {
      "check_interval_seconds": 30,
      "unrealized_loss_alert_percentage": 0.015,
      "time_based_exit_hours": 24
    },
    
    "performance_tracking": {
      "calculate_metrics_interval_minutes": 5,
      "performance_alert_thresholds": {
        "daily_loss": -0.03,
        "win_rate_below": 0.4,
        "profit_factor_below": 1.0
      }
    }
  }
}
```

### Project Configuration (`configs/project_config.json`)

```json
{
  "project": {
    "name": "ARIA ELITE",
    "version": "2.0.0",
    "environment": "production",
    "description": "Institutional-Grade AI-Powered Trading System"
  },
  
  "system": {
    "timezone": "UTC",
    "base_currency": "USD",
    "decimal_precision": 5,
    "update_interval_seconds": 60
  },
  
  "ai_models": {
    "model_directory": "./backend/models",
    "training_data_directory": "./data/training",
    "model_update_interval_hours": 24,
    "performance_threshold": 0.65,
    "auto_retrain": true,
    "max_training_samples": 100000
  },
  
  "data_sources": {
    "primary_data_provider": "yahoo_finance",
    "backup_data_providers": ["alpha_vantage"],
    "data_update_interval_seconds": 60,
    "historical_data_days": 365,
    "cache_data": true
  },
  
  "notifications": {
    "discord": {
      "enabled": true,
      "error_notifications": true,
      "trade_notifications": true,
      "system_status_notifications": true,
      "performance_reports": true
    },
    "email": {
      "enabled": false,
      "smtp_server": "",
      "smtp_port": 587,
      "username": "",
      "password": ""
    }
  },
  
  "security": {
    "api_rate_limiting": true,
    "request_logging": true,
    "ip_whitelist": [],
    "encrypt_sensitive_data": true,
    "audit_trail": true
  },
  
  "performance": {
    "database_connection_pool_size": 20,
    "max_concurrent_requests": 100,
    "cache_timeout_seconds": 300,
    "memory_optimization": true
  }
}
```

## 🤖 AI Model Configuration

### Model Parameters (`backend/models/model_config.json`)

```json
{
  "models": {
    "lightgbm": {
      "enabled": true,
      "parameters": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.2
      },
      "training": {
        "num_rounds": 1000,
        "early_stopping_rounds": 100,
        "validation_split": 0.2,
        "cross_validation_folds": 5
      }
    },
    
    "cnn": {
      "enabled": true,
      "architecture": {
        "input_shape": [100, 5],
        "conv_layers": [
          { "filters": 32, "kernel_size": 3, "activation": "relu" },
          { "filters": 64, "kernel_size": 3, "activation": "relu" },
          { "filters": 128, "kernel_size": 3, "activation": "relu" }
        ],
        "dense_layers": [
          { "units": 100, "activation": "relu", "dropout": 0.3 },
          { "units": 50, "activation": "relu", "dropout": 0.2 },
          { "units": 1, "activation": "linear" }
        ]
      },
      "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "mse",
        "metrics": ["mae"],
        "validation_split": 0.2,
        "early_stopping_patience": 10
      }
    },
    
    "mobilenet": {
      "enabled": true,
      "architecture": {
        "input_shape": [100, 5],
        "alpha": 0.5,
        "depth_multiplier": 1,
        "dropout": 0.2,
        "include_top": false,
        "pooling": "avg"
      },
      "training": {
        "epochs": 80,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "optimizer": "rmsprop",
        "loss": "mse",
        "fine_tune_layers": 10
      }
    }
  },
  
  "ensemble": {
    "enabled": true,
    "method": "weighted_average",
    "weights": {
      "lightgbm": 0.4,
      "cnn": 0.35,
      "mobilenet": 0.25
    },
    "confidence_threshold": 0.7,
    "disagreement_threshold": 0.3
  },
  
  "features": {
    "technical_indicators": [
      "sma_20", "sma_50", "sma_200",
      "ema_12", "ema_26", "ema_9",
      "rsi_14", "rsi_21",
      "macd", "macd_signal", "macd_histogram",
      "bb_upper", "bb_middle", "bb_lower",
      "atr_14", "atr_21",
      "stoch_k", "stoch_d",
      "cci_20", "williams_r"
    ],
    "price_features": [
      "open", "high", "low", "close", "volume",
      "price_change", "price_change_pct",
      "high_low_ratio", "volume_price_trend"
    ],
    "time_features": [
      "hour_of_day", "day_of_week", "day_of_month",
      "is_weekend", "is_trading_session"
    ]
  }
}
```

## 🔍 Monitoring Configuration

### Logging Configuration (`configs/logging_config.json`)

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  
  "formatters": {
    "standard": {
      "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "detailed": {
      "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
      "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
    }
  },
  
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "standard",
      "stream": "ext://sys.stdout"
    },
    
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "./logs/aria_elite.log",
      "maxBytes": 10485760,
      "backupCount": 10
    },
    
    "error_file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "ERROR",
      "formatter": "detailed",
      "filename": "./logs/aria_elite_errors.log",
      "maxBytes": 5242880,
      "backupCount": 5
    },
    
    "trading_file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "INFO",
      "formatter": "json",
      "filename": "./logs/trading_activity.log",
      "maxBytes": 20971520,
      "backupCount": 20
    }
  },
  
  "loggers": {
    "backend.core.signal_manager": {
      "level": "DEBUG",
      "handlers": ["console", "file", "trading_file"],
      "propagate": false
    },
    
    "backend.core.execution_engine": {
      "level": "DEBUG", 
      "handlers": ["console", "file", "trading_file"],
      "propagate": false
    },
    
    "backend.core.gemini_workflow_agent": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    },
    
    "backend.core.mt5_bridge": {
      "level": "DEBUG",
      "handlers": ["console", "file", "trading_file"],
      "propagate": false
    }
  },
  
  "root": {
    "level": "INFO",
    "handlers": ["console", "file", "error_file"]
  }
}
```

### Alert Configuration (`configs/alerts_config.json`)

```json
{
  "alert_channels": {
    "discord": {
      "enabled": true,
      "webhook_url": "${DISCORD_WEBHOOK_URL}",
      "rate_limit_seconds": 60
    },
    "email": {
      "enabled": false,
      "smtp_config": {
        "server": "",
        "port": 587,
        "username": "",
        "password": "",
        "tls": true
      }
    }
  },
  
  "alert_types": {
    "critical_error": {
      "enabled": true,
      "channels": ["discord"],
      "priority": "high",
      "rate_limit_minutes": 5
    },
    
    "trading_signal": {
      "enabled": true,
      "channels": ["discord"],
      "priority": "medium",
      "rate_limit_minutes": 1,
      "conditions": {
        "min_confidence": 0.8
      }
    },
    
    "position_update": {
      "enabled": true,
      "channels": ["discord"],
      "priority": "low",
      "rate_limit_minutes": 10,
      "conditions": {
        "significant_pnl_change": 0.01
      }
    },
    
    "system_status": {
      "enabled": true,
      "channels": ["discord"],
      "priority": "medium",
      "schedule": "0 */4 * * *"
    },
    
    "performance_report": {
      "enabled": true,
      "channels": ["discord"],
      "priority": "low",
      "schedule": "0 0 * * *"
    }
  },
  
  "thresholds": {
    "daily_loss_alert": -0.03,
    "drawdown_alert": -0.10,
    "win_rate_alert": 0.40,
    "connection_timeout_alert": 300,
    "memory_usage_alert": 0.85,
    "cpu_usage_alert": 0.90
  }
}
```

## 🔐 Security Configuration

### Security Settings (`configs/security_config.json`)

```json
{
  "authentication": {
    "jwt": {
      "algorithm": "HS256",
      "expiration_hours": 24,
      "refresh_token_days": 7,
      "issuer": "aria-elite-trading"
    },
    
    "api_tokens": {
      "supervisor_token_required": true,
      "token_rotation_days": 30,
      "max_failed_attempts": 5,
      "lockout_duration_minutes": 15
    }
  },
  
  "encryption": {
    "database": {
      "encrypt_sensitive_fields": true,
      "encryption_algorithm": "AES-256-GCM",
      "key_rotation_days": 90
    },
    
    "api_communication": {
      "require_https": false,
      "tls_version": "1.3",
      "cipher_suites": ["TLS_AES_256_GCM_SHA384"]
    }
  },
  
  "access_control": {
    "ip_whitelist": {
      "enabled": false,
      "allowed_ips": []
    },
    
    "rate_limiting": {
      "enabled": true,
      "general_limit": "60/minute",
      "signal_generation_limit": "10/minute",
      "trade_execution_limit": "30/minute"
    },
    
    "cors": {
      "enabled": true,
      "allowed_origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
      ],
      "allowed_methods": ["GET", "POST"],
      "allowed_headers": ["*"],
      "allow_credentials": true
    }
  },
  
  "audit": {
    "log_all_requests": true,
    "log_sensitive_data": false,
    "retention_days": 90,
    "compliance_mode": false
  }
}
```

## 🚀 Deployment Configuration

### Docker Configuration

**Production Docker Compose** (`docker-compose.prod.yml`):
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=file:/app/db/custom.db
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - MT5_LOGIN=${MT5_LOGIN}
      - MT5_PASSWORD=${MT5_PASSWORD}
      - MT5_SERVER=${MT5_SERVER}
    volumes:
      - ./db:/app/db
      - ./logs:/app/logs
      - ./configs:/app/configs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
```

### Systemd Service Configuration

**Service File** (`/etc/systemd/system/aria-elite.service`):
```ini
[Unit]
Description=ARIA ELITE Trading System
After=network.target

[Service]
Type=simple
User=trading
Group=trading
WorkingDirectory=/home/trading/ARIA-ELITE
ExecStart=/home/trading/ARIA-ELITE/.venv/bin/python start_live_trading.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aria-elite

# Environment
Environment=NODE_ENV=production
EnvironmentFile=/home/trading/ARIA-ELITE/.env

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/trading/ARIA-ELITE

[Install]
WantedBy=multi-user.target
```

## 📊 Performance Tuning

### Database Optimization

**SQLite Configuration**:
```sql
-- Performance optimizations for SQLite
PRAGMA journal_mode = WAL;          -- Write-Ahead Logging
PRAGMA synchronous = NORMAL;        -- Balanced durability/performance
PRAGMA cache_size = 10000;          -- 10MB cache
PRAGMA temp_store = memory;         -- Store temp tables in memory
PRAGMA mmap_size = 268435456;       -- 256MB memory map
PRAGMA optimize;                    -- Query optimizer
```

**Prisma Configuration** (`prisma/schema.prisma`):
```prisma
generator client {
  provider = "prisma-client-js"
  previewFeatures = ["metrics", "tracing"]
}

datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}
```

### System Performance

**Performance Settings**:
```bash
# System limits
ulimit -n 65536        # File descriptor limit
ulimit -u 32768        # Process limit

# Python optimizations  
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Node.js optimizations
export NODE_OPTIONS="--max-old-space-size=2048 --optimize-for-size"
```

## 🔧 Development Configuration

### Development Environment (`.env.development`)

```bash
NODE_ENV=development
DATABASE_URL=file:./db/development.db

# Development API keys (use test accounts)
GEMINI_API_KEY=your_test_gemini_key
MT5_LOGIN=your_demo_account
MT5_PASSWORD=your_demo_password
MT5_SERVER=your_demo_server

# Development tokens (less secure for development)
SUPERVISOR_API_TOKEN=dev_supervisor_token
JWT_SECRET=dev_jwt_secret

# Development logging
LOG_LEVEL=debug
LOG_FILE_PATH=./logs/dev

# Development Discord (optional)
DISCORD_BOT_TOKEN=your_dev_bot_token
DISCORD_WEBHOOK_URL=your_dev_webhook
```

### VS Code Configuration (`.vscode/settings.json`)

```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "typescript.preferences.importModuleSpecifier": "relative",
  "eslint.workingDirectories": ["./src"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/node_modules": true,
    "**/.next": true
  }
}
```

---

**📞 Configuration Support**: For configuration questions and customization help, refer to the main documentation or contact the development team.
