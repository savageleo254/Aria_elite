from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

from backend.core.browser_ai_agent import BrowserAIManager, browser_ai_manager
from backend.core.signal_manager import SignalManager
from backend.core.execution_engine import ExecutionEngine
from backend.core.smc_module import SMCModule
from backend.core.news_scraper import NewsScraper
from backend.core.market_microstructure import MarketMicrostructure
from backend.models.ai_models import AIModelManager
from backend.utils.config_loader import ConfigLoader
from backend.utils.logger import setup_logger
from backend.app.microstructure_routes import router as microstructure_router

load_dotenv()

app = FastAPI(
    title="ARIA Trading System API",
    description="AI-powered trading system with multi-AI browser automation (ChatGPT, Grok, Gemini, Claude)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include microstructure router
app.include_router(microstructure_router)

security = HTTPBearer()
logger = setup_logger(__name__)

config = ConfigLoader()

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str
    strategy: str
    parameters: Optional[Dict[str, Any]] = {}

class SignalResponse(BaseModel):
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    strategy: str

class TradeRequest(BaseModel):
    signal_id: str
    volume: float
    order_type: str = "market"
    slippage_tolerance: float = 0.0005

class TradeResponse(BaseModel):
    trade_id: str
    signal_id: str
    symbol: str
    volume: float
    entry_price: float
    status: str
    timestamp: datetime

class StatusResponse(BaseModel):
    system_status: str
    active_positions: int
    daily_pnl: float
    total_trades: int
    win_rate: float
    last_signal_time: Optional[datetime]
    ai_status: str

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != os.getenv("SUPERVISOR_API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

@app.on_event("startup")
async def startup_event():
    logger.info("Starting ARIA Trading System API")
    
    app.state.browser_ai_manager = browser_ai_manager
    app.state.signal_manager = SignalManager()
    app.state.execution_engine = ExecutionEngine()
    app.state.smc_module = SMCModule()
    app.state.news_scraper = NewsScraper()
    app.state.ai_model_manager = AIModelManager()
    
    # Initialize Browser AI Manager with error handling
    try:
        await app.state.browser_ai_manager.initialize()
        logger.info("Browser AI Manager initialized successfully")
    except Exception as e:
        logger.warning(f"Browser AI Manager initialization failed: {e}")
        logger.info("Server will continue with limited AI capabilities")
    
    await app.state.execution_engine.initialize()
    
    logger.info("ARIA Trading System API started successfully")

@app.get("/")
async def root():
    return {"message": "ARIA Trading System API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/status", response_model=StatusResponse)
async def get_status(token: str = Depends(verify_token)):
    try:
        status = await app.state.signal_manager.get_system_status()
        # Get AI manager status
        ai_status = "Unknown"
        if hasattr(app.state, 'browser_ai_manager'):
            try:
                ai_status = "Multi-AI Ready"
                ai_system_status = await app.state.browser_ai_manager.get_system_status()
                if ai_system_status['browser_ai_manager']['initialized']:
                    ai_status = f"Multi-AI Active ({len(ai_system_status['browser_ai_manager']['providers_status'])} providers)"
                else:
                    ai_status = "Multi-AI Initializing"
            except Exception as e:
                ai_status = f"Multi-AI Error: {str(e)}"
        
        status['ai_status'] = ai_status
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/status")
async def get_ai_status(token: str = Depends(verify_token)):
    """Get detailed AI system status"""
    try:
        if hasattr(app.state, 'browser_ai_manager'):
            ai_status = await app.state.browser_ai_manager.get_system_status()
            return ai_status
        else:
            return {"error": "AI Manager not initialized"}
    except Exception as e:
        logger.error(f"Error getting AI status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/accounts")
async def add_ai_account(
    provider: str,
    email: str,
    password: str,
    max_daily_usage: int = 100,
    token: str = Depends(verify_token)
):
    """Add new AI account"""
    try:
        from backend.core.browser_ai_agent import AIProvider
        
        ai_provider = AIProvider(provider.lower())
        success = await app.state.browser_ai_manager.add_account(
            ai_provider, email, password, max_daily_usage
        )
        
        if success:
            return {"message": f"Successfully added {provider} account"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add account")
            
    except Exception as e:
        logger.error(f"Error adding AI account: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signals/generate", response_model=SignalResponse)
async def generate_signal(request: SignalRequest, token: str = Depends(verify_token)):
    try:
        signal = await app.state.browser_ai_manager.generate_signal(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            parameters=request.parameters
        )
        return SignalResponse(**signal)
    except Exception as e:
        logger.error(f"Error generating signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trades/execute", response_model=TradeResponse)
async def execute_trade(request: TradeRequest, token: str = Depends(verify_token)):
    try:
        trade = await app.state.execution_engine.execute_trade(
            signal_id=request.signal_id,
            volume=request.volume,
            order_type=request.order_type,
            slippage_tolerance=request.slippage_tolerance
        )
        return TradeResponse(**trade)
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/active")
async def get_active_trades(token: str = Depends(verify_token)):
    try:
        trades = await app.state.execution_engine.get_active_trades()
        return {"active_trades": trades}
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/history")
async def get_trade_history(
    limit: int = 100,
    offset: int = 0,
    token: str = Depends(verify_token)
):
    try:
        history = await app.state.execution_engine.get_trade_history(limit, offset)
        return {"trade_history": history}
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/pause")
async def pause_system(token: str = Depends(verify_token)):
    try:
        await app.state.signal_manager.pause()
        return {"message": "System paused successfully"}
    except Exception as e:
        logger.error(f"Error pausing system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/resume")
async def resume_system(token: str = Depends(verify_token)):
    try:
        await app.state.signal_manager.resume()
        return {"message": "System resumed successfully"}
    except Exception as e:
        logger.error(f"Error resuming system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/kill-switch")
async def activate_kill_switch(token: str = Depends(verify_token)):
    try:
        await app.state.execution_engine.emergency_close_all()
        await app.state.signal_manager.pause()
        return {"message": "Kill switch activated successfully"}
    except Exception as e:
        logger.error(f"Error activating kill switch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_model_status(token: str = Depends(verify_token)):
    try:
        status = await app.state.ai_model_manager.get_model_status()
        return {"model_status": status}
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks, token: str = Depends(verify_token)):
    try:
        background_tasks.add_task(app.state.ai_model_manager.retrain_all_models)
        return {"message": "Model retraining started in background"}
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/sentiment")
async def get_news_sentiment(symbol: Optional[str] = None, token: str = Depends(verify_token)):
    try:
        sentiment = await app.state.news_scraper.get_sentiment_analysis(symbol)
        return {"sentiment_analysis": sentiment}
    except Exception as e:
        logger.error(f"Error getting news sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/performance")
async def get_performance_metrics(
    period: str = "1d",
    token: str = Depends(verify_token)
):
    try:
        metrics = await app.state.signal_manager.get_performance_metrics(period)
        return {"performance_metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
