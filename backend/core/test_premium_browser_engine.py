"""
Test Suite for Premium Browser Automation Engine
ARIA-DAN Institutional Trading System
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from backend.core.premium_browser_engine import PremiumBrowserEngine, BrowserSession, PremiumModelTier, TaskComplexity

# Add root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

@pytest.fixture
def mock_browser_session():
    session = BrowserSession("gemini", "test@example.com", "password")
    session.driver = MagicMock()
    return session

@pytest.mark.asyncio
async def test_json_extraction():
    """Test robust JSON extraction with various response formats"""
    engine = PremiumBrowserEngine()
    
    # Test pure JSON response
    response1 = '{"signal": "BUY", "confidence": 0.85}'
    result1 = engine._extract_json_response(response1)
    assert result1["signal"] == "BUY"
    assert result1["confidence"] == 0.85
    
    # Test wrapped JSON response
    response2 = 'Here is my analysis: {"signal": "SELL", "confidence": 0.92}'
    result2 = engine._extract_json_response(response2)
    assert result2["signal"] == "SELL"
    assert result2["confidence"] == 0.92
    
    # Test invalid JSON fallback
    response3 = 'This is not valid JSON'
    result3 = engine._extract_json_response(response3)
    assert result3["signal"] == "HOLD"
    assert result3["confidence"] == 0.5
    assert result3["reasoning"] == 'This is not valid JSON'
    
    # Test partial JSON
    response4 = '{"signal": "BUY", "confidence": 0.78, "reasoning": "Market looks strong"'
    result4 = engine._extract_json_response(response4)
    assert result4["signal"] == "HOLD"
    assert result4["confidence"] == 0.5
    assert "Market looks strong" in result4["reasoning"]

@pytest.mark.asyncio
@patch('backend.core.premium_browser_engine.pickle')
async def test_session_persistence(mock_pickle, mock_browser_session):
    """Test cookie loading during session initialization"""
    mock_pickle.load.return_value = [{'name': 'test_cookie', 'value': '123'}]
    
    # Simulate existing cookie file
    with patch("os.path.exists", return_value=True):
        await mock_browser_session.initialize()
        
        # Verify cookies added
        mock_browser_session.driver.add_cookie.assert_called_with({'name': 'test_cookie', 'value': '123'})

@pytest.mark.asyncio
@patch('backend.core.premium_browser_engine.uc')
async def test_gemini_automation(mock_uc):
    """Test Gemini query execution flow"""
    engine = PremiumBrowserEngine()
    
    with patch.object(engine, '_gemini_login', return_value=True) as mock_login,\
         patch.object(engine, '_execute_gemini_query', return_value='{"signal": "BUY"}') as mock_query:
        
        # Execute Gemini query
        result = await engine.execute_premium_query(
            "Test prompt",
            PremiumModelTier.GEMINI_2_5_PRO,
            TaskComplexity.CRITICAL
        )
        
        assert result == {"signal": "BUY"}
        mock_login.assert_called_once()
        mock_query.assert_called_once()

# Add similar tests for Claude and Grok

if __name__ == "__main__":
    pytest.main(["-v", "test_premium_browser_engine.py"])
