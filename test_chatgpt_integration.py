import asyncio
from backend.core.premium_browser_engine import PremiumBrowserEngine, TaskComplexity

async def test_chatgpt():
    engine = PremiumBrowserEngine()
    await engine.initialize()
    
    # Use the new account
    symbol = "AAPL"
    timeframe = "1h"
    market_context = {
        "current_price": 150.0,
        "trend": "bullish",
        "volatility": 0.02,
        "rsi": 60.0,
        "volume": 1000000
    }
    complexity = TaskComplexity.COMPLEX
    
    signal = await engine.generate_signal(symbol, timeframe, market_context, complexity)
    
    # Explicit cleanup
    await engine.active_sessions["chatgpt"].cleanup()
    
    # Print signal without .value access
    print(f"Signal: {signal}")

if __name__ == "__main__":
    asyncio.run(test_chatgpt())
