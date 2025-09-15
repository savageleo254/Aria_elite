# ARIA Trading System

A comprehensive AI-powered trading system built with Next.js 15, FastAPI, and Gemini AI. This system provides institutional-grade forex trading capabilities with smart money concepts analysis, ensemble AI models, and robust risk management.

## 🚀 Features

### Core Architecture
- **Gemini AI Integration**: Advanced AI workflow orchestration using Google's Gemini as the primary decision-making brain
- **FastAPI Backend**: High-performance Python backend with typed endpoints and async processing
- **React Frontend**: Modern, responsive TypeScript interface with real-time updates
- **Microservices Design**: Modular architecture with independent services for scalability

### Trading Capabilities
- **Smart Money Concepts (SMC)**: Advanced market structure analysis including:
  - Break of structure detection
  - Liquidity zone identification
  - Order block analysis
  - Fair value gaps (FVG)
  - Change of character (CHOCH) events

- **AI Ensemble Models**: Multiple machine learning models working together:
  - TinyML LSTM for time series prediction
  - LightGBM for gradient boosting
  - 1D-CNN for pattern recognition
  - MobileNetV3 for feature extraction

- **News Sentiment Analysis**: Real-time news scraping and sentiment analysis with impact scoring

### Risk Management
- **Position Limits**: Configurable maximum open positions
- **Risk per Trade**: Dynamic risk calculation based on account size
- **Stop Loss/Take Profit**: Automated risk management with ATR-based calculations
- **Kill Switch**: Emergency system shutdown with manual override capabilities
- **Drawdown Control**: Maximum drawdown protection with automatic position closure

### System Monitoring
- **Supervisor API**: Comprehensive system health monitoring
- **Real-time Metrics**: CPU, memory, disk usage tracking
- **Service Health**: Individual service status monitoring
- **Alert System**: Multi-channel notifications (Discord, email)
- **Performance Analytics**: Detailed trading statistics and performance metrics

### Backtesting & Optimization
- **Historical Data Analysis**: Comprehensive backtesting engine with multiple strategies
- **Parameter Optimization**: Walk-forward optimization with Monte Carlo simulation
- **Strategy Comparison**: Side-by-side performance analysis
- **Risk Metrics**: Sharpe ratio, profit factor, maximum drawdown calculations

## 🏗️ Project Structure

```
ARIA-Rebuild-With-Gemini/
├── backend/                    # FastAPI backend application
│   ├── app/                  # Main application code
│   │   ├── main.py          # FastAPI app entry point
│   │   └── __init__.py
│   ├── core/                 # Core trading modules
│   │   ├── gemini_workflow_agent.py
│   │   ├── signal_manager.py
│   │   ├── execution_engine.py
│   │   ├── smc_module.py
│   │   ├── news_scraper.py
│   │   └── backtesting_engine.py
│   ├── models/               # AI model implementations
│   │   └── ai_models.py
│   └── utils/                # Utility functions
│       ├── config_loader.py
│       └── logger.py
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/           # Page components
│   │   ├── contexts/        # React contexts
│   │   ├── hooks/           # Custom hooks
│   │   ├── types/           # TypeScript types
│   │   └── utils/           # Utility functions
│   └── public/              # Static assets
├── supervisor/               # System monitoring service
│   └── supervisor.py
├── discord_agent/           # Discord bot for manual control
│   └── discord_bot.py
├── models/                   # AI model training and artifacts
│   ├── training/            # Training scripts
│   ├── evaluation/          # Model evaluation
│   └── artifacts/           # Trained models
├── configs/                  # Configuration files
│   ├── project_config.json
│   ├── strategy_config.json
│   └── execution_config.json
├── logs/                     # System logs and trade history
│   ├── trades/              # Trade history
│   ├── strategies/          # Strategy outcomes
│   └── system/              # System logs
├── ci/                       # CI/CD workflows
├── tests/                    # Test suites
└── docs/                     # Documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- Docker (optional)
- MetaTrader 5 (for live trading)

### Environment Variables
Create a `.env` file in the project root:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
SUPERVISOR_API_TOKEN=your_supervisor_token_here

# MT5 Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# Discord Bot
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_ALLOWED_USERS=user_id1,user_id2,user_id3
```

### Backend Setup

1. **Navigate to backend directory**:
```bash
cd backend
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Initialize database**:
```bash
npm run db:push
```

4. **Start the backend server**:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install Node.js dependencies**:
```bash
npm install
```

3. **Start the development server**:
```bash
npm start
```

### Supervisor Setup

1. **Navigate to supervisor directory**:
```bash
cd supervisor
```

2. **Start the supervisor service**:
```bash
python supervisor.py
```

### Discord Bot Setup

1. **Navigate to discord_agent directory**:
```bash
cd discord_agent
```

2. **Start the Discord bot**:
```bash
python discord_bot.py
```

## 📊 Usage

### Dashboard
- **System Status**: Real-time monitoring of system health and performance
- **Active Positions**: View and manage currently open trades
- **Recent Alerts**: System notifications and warnings
- **Performance Metrics**: Win rate, P&L, and trading statistics

### Trading Control
- **System Controls**: Pause/resume trading, activate kill switch
- **Position Management**: View and close active positions
- **Risk Settings**: Configure risk parameters and limits

### Analytics
- **Performance Charts**: P&L curves, volume analysis, and strategy breakdowns
- **Risk Metrics**: Sharpe ratio, maximum drawdown, profit factor
- **Symbol Performance**: Performance analysis by trading symbol

### AI Models
- **Model Status**: Monitor training status and performance
- **Ensemble Weights**: Configure AI model contributions
- **Retraining**: Schedule and execute model retraining

### News & Sentiment
- **News Feed**: Real-time forex news from multiple sources
- **Sentiment Analysis**: Market sentiment scoring and impact analysis
- **Symbol Sentiment**: Sentiment breakdown by currency pair

### Backtesting
- **Strategy Testing**: Test strategies with historical data
- **Parameter Optimization**: Find optimal strategy parameters
- **Performance Comparison**: Compare multiple strategies side-by-side

### Settings
- **Trading Parameters**: Configure position limits, risk settings
- **Execution Settings**: Slippage tolerance, retry attempts
- **AI Settings**: Model confidence thresholds, ensemble weights
- **Risk Management**: Drawdown limits, loss prevention

## 🔧 Configuration

### Strategy Configuration
Edit `configs/strategy_config.json` to customize trading strategies:

```json
{
  "strategies": {
    "smc_strategy": {
      "enabled": true,
      "parameters": {
        "min_confidence": 0.75,
        "max_risk_per_trade": 0.02,
        "reward_ratio": 2.0
      }
    },
    "ai_strategy": {
      "enabled": true,
      "model_weights": {
        "tinyml_lstm": 0.3,
        "lightgbm": 0.3,
        "1d_cnn": 0.2,
        "mobilenetv3": 0.2
      }
    }
  }
}
```

### Risk Management
Configure risk parameters in the settings interface or via configuration files:

```json
{
  "risk_management": {
    "max_daily_loss": 0.05,
    "max_drawdown": 0.15,
    "position_sizing": {
      "method": "fixed_fractional",
      "risk_per_trade": 0.02
    }
  }
}
```

## 🚨 Safety Features

### Kill Switch
- **Automatic Activation**: Triggers on CPU/memory overload or excessive drawdown
- **Manual Override**: Discord commands and web interface activation
- **Immediate Effect**: Closes all positions and stops all trading activity

### Risk Limits
- **Position Limits**: Maximum concurrent positions to prevent overexposure
- **Daily Loss Limits**: Automatic trading halt when daily loss threshold reached
- **Maximum Drawdown**: System shutdown when drawdown exceeds configured limit

### Monitoring
- **Health Checks**: Continuous monitoring of system resources and service health
- **Alert System**: Multi-channel notifications for critical events
- **Performance Tracking**: Real-time performance metrics and anomaly detection

## 🤖 Discord Commands

The Discord bot provides manual control over the trading system:

- `!status` - Get current system status
- `!pause` - Pause trading system
- `!resume` - Resume trading system
- `!kill_switch` - Activate emergency kill switch
- `!train [model]` - Train specific AI model
- `!positions` - Show active positions
- `!logs [limit]` - View system logs
- `!help` - Show available commands

## 📈 Performance Optimization

### System Requirements
- **Minimum**: Intel i5-7300U, 8GB RAM, 256GB SSD
- **Recommended**: Intel i7+, 16GB RAM, 512GB SSD
- **Network**: Stable internet connection for live trading

### Optimization Features
- **Model Pruning**: Reduced model size for faster inference
- **Quantization**: Lower precision models for improved performance
- **Caching**: Result caching to reduce redundant computations
- **Async Processing**: Non-blocking operations for improved responsiveness

## 🔒 Security

### Authentication
- **Token-based API**: Secure API access with bearer tokens
- **Environment Variables**: Sensitive data stored in environment variables
- **Role-based Access**: Different access levels for different operations

### Data Protection
- **Local Execution**: All processing happens locally, no cloud dependency
- **Encrypted Storage**: Sensitive data encrypted at rest
- **Audit Logging**: Comprehensive logging of all system actions

## 🧪 Testing

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Integration tests
npm run test:integration
```

### Test Coverage
- Unit tests for all core modules
- Integration tests for API endpoints
```

### Trading Configuration
Configure trading parameters in `configs/`:
- `strategy_config.json` - Strategy parameters and weights  
- `execution_config.json` - Risk management and execution settings
- `project_config.json` - System-wide configuration

## 🔐 Security Features

- **Token-based API Authentication** - Secure endpoint access
- **Environment Variable Management** - No hardcoded credentials
- **Database Encryption** - SQLite with secure storage
- **Input Validation** - Comprehensive request sanitization
- **Audit Logging** - Complete system activity tracking

## 📈 Monitoring & Observability

### Real-time Monitoring
- **System Dashboard** - Web-based monitoring interface
- **Discord Notifications** - Instant alerts and status updates
- **Performance Metrics** - Real-time P&L, win rates, and system health
- **Structured Logging** - Comprehensive activity tracking

### Alerting System
- **Critical Errors** - Immediate Discord notifications
- **Performance Thresholds** - Automated risk alerts
- **System Health** - Connection and component monitoring
- **Trading Alerts** - Signal generation and execution notifications

## 🚨 Emergency Procedures

### Kill Switch Activation
```bash
# Via API
curl -X POST http://localhost:8000/system/kill-switch \
  -H "Authorization: Bearer your_token"

# Via Discord (if configured)
!kill-switch
```

### System Recovery
```bash
# Restart system components
python start_live_trading.py

# Check system status
curl http://localhost:8000/status
```

## 🛠️ Development

### Project Structure
```
ARIA-ELITE/
├── backend/                 # Python FastAPI backend
│   ├── app/                # API application
│   ├── core/               # Trading engines and agents
│   ├── models/             # AI models and data structures
│   └── utils/              # Utilities and helpers
├── src/                    # Next.js frontend
│   ├── app/                # App router pages
│   ├── components/         # React components  
│   └── lib/                # Frontend utilities
├── configs/                # System configuration files
├── db/                     # SQLite database
├── logs/                   # System logs
└── scripts/                # Utility scripts
```

### Development Commands
```bash
# Frontend development
npm run dev

# Backend development  
cd backend && python -m uvicorn app.main:app --reload

# Database operations
npm run db:generate      # Generate Prisma client
npm run db:push         # Push schema changes
npm run db:migrate      # Create migration

# Testing
pytest backend/tests/
npm run test
```

## 📄 License

Proprietary institutional trading system. All rights reserved.

## ⚠️ Risk Disclaimer

This is a live trading system that executes real trades with real money. Always:
- Test thoroughly in paper trading mode first
- Start with minimal position sizes
- Monitor the system continuously during initial deployment
- Have emergency procedures ready
- Understand the risks involved in automated trading

**Trading involves substantial risk of loss and is not suitable for all investors.**

## 📞 Support

For support and questions:
- **Documentation**: Check the `/docs` directory
- **Issues**: Open an issue on GitHub
- **Discord**: Join our community server
- **Email**: Contact the development team

---

**Disclaimer**: This is a complex trading system that involves real financial risk. Use at your own risk and ensure proper testing before deploying with real funds. Past performance does not guarantee future results.