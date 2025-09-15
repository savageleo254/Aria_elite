import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import feedparser
from bs4 import BeautifulSoup
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class NewsScraper:
    """
    Scrapes forex news from multiple sources and performs sentiment analysis
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.forexfactory.com/rss.php",
            "https://www.investing.com/rss/news.rss"
        ]
        self.news_cache = []
        self.sentiment_scores = {}
        
    async def initialize(self):
        """Initialize the news scraper"""
        try:
            logger.info("Initializing News Scraper")
            
            # Load configuration
            await self._load_config()
            
            # Start news collection loop
            asyncio.create_task(self._news_collection_loop())
            
            self.is_initialized = True
            logger.info("News Scraper initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize News Scraper: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load configuration for news scraping"""
        try:
            self.project_config = self.config.load_project_config()
            self.news_config = self.project_config.get("news", {})
            self.update_interval = self.news_config.get("update_interval", 300)  # 5 minutes
            
            logger.info("News configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load news configuration: {str(e)}")
            raise
    
    async def _news_collection_loop(self):
        """Continuously collect news from various sources"""
        while True:
            try:
                await self._collect_news()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in news collection loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _collect_news(self):
        """Collect news from all configured sources"""
        try:
            logger.info("Collecting news from sources")
            
            for source in self.news_sources:
                try:
                    news_items = await self._scrape_source(source)
                    self.news_cache.extend(news_items)
                except Exception as e:
                    logger.warning(f"Failed to scrape {source}: {str(e)}")
            
            # Keep only recent news (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.news_cache = [
                item for item in self.news_cache 
                if item.get("timestamp", datetime.min) > cutoff_time
            ]
            
            # Perform sentiment analysis
            await self._analyze_sentiment()
            
            logger.info(f"Collected {len(self.news_cache)} news items")
            
        except Exception as e:
            logger.error(f"Error collecting news: {str(e)}")
    
    async def _scrape_source(self, source_url: str) -> List[Dict[str, Any]]:
        """Scrape news from a specific source"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        news_items = []
                        for entry in feed.entries[:10]:  # Limit to 10 items per source
                            news_item = {
                                "title": entry.get("title", ""),
                                "summary": entry.get("summary", ""),
                                "link": entry.get("link", ""),
                                "timestamp": datetime.now(),
                                "source": source_url
                            }
                            news_items.append(news_item)
                        
                        return news_items
                    else:
                        logger.warning(f"Failed to fetch from {source_url}: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Error scraping {source_url}: {str(e)}")
            return []
    
    async def _analyze_sentiment(self):
        """Analyze sentiment of collected news"""
        try:
            for item in self.news_cache:
                sentiment = await self._calculate_sentiment(item)
                item["sentiment"] = sentiment
                
                # Update symbol-specific sentiment
                symbol = self._extract_symbol(item["title"] + " " + item["summary"])
                if symbol:
                    if symbol not in self.sentiment_scores:
                        self.sentiment_scores[symbol] = []
                    self.sentiment_scores[symbol].append(sentiment)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
    
    async def _calculate_sentiment(self, news_item: Dict[str, Any]) -> float:
        """Calculate sentiment score for a news item"""
        try:
            text = news_item["title"] + " " + news_item["summary"]
            
            # Simple sentiment analysis based on keywords
            positive_words = [
                "bullish", "rise", "gain", "up", "positive", "strong", "growth",
                "increase", "boost", "surge", "rally", "breakthrough", "recovery"
            ]
            negative_words = [
                "bearish", "fall", "drop", "down", "negative", "weak", "decline",
                "decrease", "crash", "plunge", "slump", "recession", "crisis"
            ]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count + negative_count == 0:
                return 0.0  # Neutral
            
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            return max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1
                    
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return 0.0
    
    def _extract_symbol(self, text: str) -> Optional[str]:
        """Extract currency pair symbols from text"""
        try:
            # Common forex pairs
            symbols = [
                "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
                "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "AUDJPY",
                "EURAUD", "EURCHF", "GBPCHF", "AUDCAD", "AUDCHF", "CADCHF"
            ]
            
            text_upper = text.upper()
            for symbol in symbols:
                if symbol in text_upper:
                    return symbol
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting symbol: {str(e)}")
            return None
    
    async def get_sentiment_analysis(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get sentiment analysis for a specific symbol or overall"""
        try:
            if symbol:
                if symbol in self.sentiment_scores:
                    scores = self.sentiment_scores[symbol]
                    avg_sentiment = sum(scores) / len(scores)
                    return {
                        "symbol": symbol,
                        "sentiment": avg_sentiment,
                        "count": len(scores),
                        "timestamp": datetime.now()
                    }
                else:
                    return {
                        "symbol": symbol,
                        "sentiment": 0.0,
                        "count": 0,
                        "timestamp": datetime.now()
                    }
            else:
                # Overall sentiment
                all_scores = []
                for symbol_scores in self.sentiment_scores.values():
                    all_scores.extend(symbol_scores)
                
                if all_scores:
                    avg_sentiment = sum(all_scores) / len(all_scores)
                    return {
                        "overall_sentiment": avg_sentiment,
                        "total_news_items": len(self.news_cache),
                        "symbols_analyzed": len(self.sentiment_scores),
                        "timestamp": datetime.now()
                    }
                else:
                    return {
                        "overall_sentiment": 0.0,
                        "total_news_items": 0,
                        "symbols_analyzed": 0,
                        "timestamp": datetime.now()
                    }
                
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {str(e)}")
            return {}
    
    async def get_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest news items"""
        try:
            # Sort by timestamp and return latest items
            sorted_news = sorted(
                self.news_cache, 
                key=lambda x: x.get("timestamp", datetime.min), 
                reverse=True
            )
            return sorted_news[:limit]
            
        except Exception as e:
            logger.error(f"Error getting latest news: {str(e)}")
            return []
    
    async def search_news(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search news items by query"""
        try:
            query_lower = query.lower()
            matching_news = []
            
            for item in self.news_cache:
                if (query_lower in item["title"].lower() or 
                    query_lower in item["summary"].lower()):
                    matching_news.append(item)
            
            # Sort by timestamp and return latest matches
            sorted_news = sorted(
                matching_news,
                key=lambda x: x.get("timestamp", datetime.min),
                reverse=True
            )
            return sorted_news[:limit]
            
        except Exception as e:
            logger.error(f"Error searching news: {str(e)}")
            return []