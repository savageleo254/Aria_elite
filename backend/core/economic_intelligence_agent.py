import asyncio
import pandas as pd
import numpy as np
import logging
import json
import re
import sqlite3
import aiohttp
import websockets
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import feedparser
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from collections import deque, defaultdict

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

# ARIA-DAN Wall Street Domination Economic Intelligence Engine
# Production-grade real-time economic data processing

class ImpactLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EconomicEvent:
    timestamp: datetime
    currency: str
    event_name: str
    impact_level: ImpactLevel
    actual_value: Optional[float]
    forecast_value: Optional[float]
    previous_value: Optional[float]
    sentiment_score: float
    volatility_prediction: float
    event_id: str
    source: str
    confidence: float
    market_impact_score: float

@dataclass
class NewsItem:
    headline: str
    content: str
    timestamp: datetime
    source: str
    currency_tags: List[str]
    sentiment_score: float
    impact_score: float
    keywords: List[str]
    news_id: str
    
@dataclass
class CentralBankStatement:
    bank: str
    currency: str
    statement: str
    timestamp: datetime
    hawkish_dovish_score: float
    policy_change_probability: float
    key_phrases: List[str]
    speaker: str

class EconomicIntelligenceAgent:
    """Economic Intelligence Layer for ARIA-DAN"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.economic_events = []
        self.sentiment_data = {}
        self.volatility_predictions = {}
        
        # Impact thresholds
        self.impact_thresholds = {
            'high_impact_deviation': 0.3,  # 30% deviation from forecast
            'volatility_spike_threshold': 2.0,  # 2x normal volatility
            'sentiment_extreme_threshold': 0.8  # ±0.8 sentiment score
        }
        
    async def initialize(self):
        try:
            logger.info("Initializing Economic Intelligence Agent")
            
            # Start monitoring loops
            self.news_task = asyncio.create_task(self._news_monitoring_loop())
            self.event_task = asyncio.create_task(self._economic_event_loop())
            
            self.is_initialized = True
            logger.info("Economic Intelligence Agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Economic Intelligence Agent: {str(e)}")
            raise
    
    async def _news_monitoring_loop(self):
        """Monitor economic news and central bank communications"""
        while self.is_initialized:
            try:
                await self._process_economic_news()
                await self._analyze_central_bank_communications()
                await asyncio.sleep(300)  # 5-minute cycles
                
            except Exception as e:
                logger.error(f"Error in news monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _economic_event_loop(self):
        """Process economic calendar events"""
        while self.is_initialized:
            try:
                await self._process_economic_calendar()
                await self._predict_event_impact()
                await asyncio.sleep(900)  # 15-minute cycles
                
            except Exception as e:
                logger.error(f"Error in economic event loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_economic_news(self):
        """Process and analyze economic news"""
        try:
            # Simulate news processing
            news_items = await self._fetch_economic_news()
            
            for news in news_items:
                sentiment = self._analyze_news_sentiment(news)
                impact_prediction = self._predict_news_impact(news, sentiment)
                
                # Store sentiment data
                currency = news.get('currency', 'USD')
                if currency not in self.sentiment_data:
                    self.sentiment_data[currency] = []
                
                self.sentiment_data[currency].append({
                    'timestamp': datetime.now(),
                    'sentiment': sentiment,
                    'impact_prediction': impact_prediction,
                    'headline': news.get('headline', '')
                })
                
                # Keep only last 1000 items per currency
                if len(self.sentiment_data[currency]) > 1000:
                    self.sentiment_data[currency] = self.sentiment_data[currency][-1000:]
            
        except Exception as e:
            logger.error(f"Error processing economic news: {str(e)}")
    
    async def _fetch_economic_news(self) -> List[Dict[str, Any]]:
        """Fetch live economic news from ForexFactory and free sources"""
        try:
            news_items = []
            
            # Fetch from ForexFactory RSS
            ff_news = await self._fetch_forexfactory_news()
            news_items.extend(ff_news)
            
            # Fetch from other free sources
            reuters_news = await self._fetch_reuters_forex_news()
            news_items.extend(reuters_news)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching live economic news: {str(e)}")
            return []
    
    async def _fetch_forexfactory_news(self) -> List[Dict[str, Any]]:
        """Fetch news from ForexFactory"""
        try:
            async with aiohttp.ClientSession() as session:
                # ForexFactory news RSS feed
                url = "https://www.forexfactory.com/rss.xml"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        news_items = []
                        for entry in feed.entries[:20]:  # Get latest 20 items
                            # Extract currency from title/description
                            currencies = self._extract_currencies(entry.title + ' ' + entry.get('description', ''))
                            
                            news_item = {
                                'headline': entry.title,
                                'currency': currencies[0] if currencies else 'USD',
                                'currencies': currencies,
                                'timestamp': datetime.fromtimestamp(entry.published_parsed[0] if hasattr(entry, 'published_parsed') else datetime.now().timestamp()),
                                'content': entry.get('description', ''),
                                'source': 'ForexFactory',
                                'link': entry.get('link', '')
                            }
                            news_items.append(news_item)
                        
                        logger.info(f"Fetched {len(news_items)} news items from ForexFactory")
                        return news_items
                    else:
                        logger.warning(f"ForexFactory RSS returned status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching ForexFactory news: {str(e)}")
            return []
    
    async def _fetch_reuters_forex_news(self) -> List[Dict[str, Any]]:
        """Fetch forex news from Reuters free feed"""
        try:
            async with aiohttp.ClientSession() as session:
                # Reuters forex RSS feed (free)
                url = "https://feeds.reuters.com/reuters/businessNews"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        forex_news = []
                        for entry in feed.entries[:15]:  # Get latest 15 items
                            title_content = entry.title + ' ' + entry.get('description', '')
                            
                            # Filter for forex-related news
                            if self._is_forex_related(title_content):
                                currencies = self._extract_currencies(title_content)
                                
                                news_item = {
                                    'headline': entry.title,
                                    'currency': currencies[0] if currencies else 'USD',
                                    'currencies': currencies,
                                    'timestamp': datetime.fromtimestamp(entry.published_parsed[0] if hasattr(entry, 'published_parsed') else datetime.now().timestamp()),
                                    'content': entry.get('description', ''),
                                    'source': 'Reuters',
                                    'link': entry.get('link', '')
                                }
                                forex_news.append(news_item)
                        
                        logger.info(f"Fetched {len(forex_news)} forex news items from Reuters")
                        return forex_news
                    else:
                        logger.warning(f"Reuters RSS returned status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Reuters news: {str(e)}")
            return []
    
    def _analyze_news_sentiment(self, news: Dict[str, Any]) -> float:
        """Analyze sentiment of economic news"""
        try:
            text = news.get('headline', '') + ' ' + news.get('content', '')
            blob = TextBlob(text)
            return float(blob.sentiment.polarity)  # -1 to 1 scale
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return 0.0
    
    def _predict_news_impact(self, news: Dict[str, Any], sentiment: float) -> float:
        """Predict market impact of news"""
        try:
            # Simple impact prediction based on keywords and sentiment
            high_impact_keywords = ['fed', 'interest rate', 'inflation', 'gdp', 'employment']
            headline = news.get('headline', '').lower()
            
            keyword_score = sum(1 for keyword in high_impact_keywords if keyword in headline)
            impact_score = (abs(sentiment) * 0.7) + (keyword_score * 0.3)
            
            return min(impact_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error predicting news impact: {str(e)}")
            return 0.0
    
    async def _analyze_central_bank_communications(self):
        """Analyze central bank speeches and communications"""
        try:
            # Simulate central bank communication analysis
            cb_communications = await self._fetch_cb_communications()
            
            for comm in cb_communications:
                hawkish_dovish_score = self._analyze_hawkish_dovish_tone(comm)
                policy_change_probability = self._predict_policy_changes(comm)
                
                # Update policy expectations
                currency = comm.get('currency', 'USD')
                self._update_policy_expectations(currency, hawkish_dovish_score, policy_change_probability)
            
        except Exception as e:
            logger.error(f"Error analyzing central bank communications: {str(e)}")
    
    async def _fetch_cb_communications(self) -> List[Dict[str, Any]]:
        """Fetch central bank communications from live sources"""
        try:
            cb_communications = []
            
            # Fetch Fed communications
            fed_comms = await self._fetch_fed_communications()
            cb_communications.extend(fed_comms)
            
            # Fetch ECB communications  
            ecb_comms = await self._fetch_ecb_communications()
            cb_communications.extend(ecb_comms)
            
            return cb_communications
            
        except Exception as e:
            logger.error(f"Error fetching central bank communications: {str(e)}")
            return []
    
    async def _fetch_fed_communications(self) -> List[Dict[str, Any]]:
        """Fetch Federal Reserve communications"""
        try:
            async with aiohttp.ClientSession() as session:
                # Fed news RSS feed
                url = "https://www.federalreserve.gov/feeds/press_all.xml"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        fed_items = []
                        for entry in feed.entries[:10]:  # Latest 10 items
                            comm_item = {
                                'currency': 'USD',
                                'bank': 'Federal Reserve',
                                'speaker': 'Fed Official',
                                'content': entry.get('description', entry.title),
                                'headline': entry.title,
                                'timestamp': datetime.fromtimestamp(entry.published_parsed[0] if hasattr(entry, 'published_parsed') else datetime.now().timestamp()),
                                'source': 'Fed',
                                'link': entry.get('link', '')
                            }
                            fed_items.append(comm_item)
                        
                        logger.info(f"Fetched {len(fed_items)} Fed communications")
                        return fed_items
                    else:
                        logger.warning(f"Fed RSS returned status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching Fed communications: {str(e)}")
            return []
    
    async def _fetch_ecb_communications(self) -> List[Dict[str, Any]]:
        """Fetch European Central Bank communications"""
        try:
            async with aiohttp.ClientSession() as session:
                # ECB press releases RSS
                url = "https://www.ecb.europa.eu/rss/press.xml"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        ecb_items = []
                        for entry in feed.entries[:10]:
                            comm_item = {
                                'currency': 'EUR',
                                'bank': 'European Central Bank',
                                'speaker': 'ECB Official',
                                'content': entry.get('description', entry.title),
                                'headline': entry.title,
                                'timestamp': datetime.fromtimestamp(entry.published_parsed[0] if hasattr(entry, 'published_parsed') else datetime.now().timestamp()),
                                'source': 'ECB',
                                'link': entry.get('link', '')
                            }
                            ecb_items.append(comm_item)
                        
                        logger.info(f"Fetched {len(ecb_items)} ECB communications")
                        return ecb_items
                    else:
                        logger.warning(f"ECB RSS returned status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching ECB communications: {str(e)}")
            return []
    
    def _analyze_hawkish_dovish_tone(self, communication: Dict[str, Any]) -> float:
        """Analyze hawkish/dovish tone of central bank communication"""
        try:
            content = communication.get('content', '').lower()
            
            hawkish_keywords = ['inflation', 'raise rates', 'tighten', 'restrictive', 'overheating']
            dovish_keywords = ['support growth', 'accommodate', 'lower rates', 'stimulus', 'employment']
            
            hawkish_count = sum(1 for keyword in hawkish_keywords if keyword in content)
            dovish_count = sum(1 for keyword in dovish_keywords if keyword in content)
            
            # Return score: -1 (dovish) to +1 (hawkish)
            total_keywords = hawkish_count + dovish_count
            if total_keywords == 0:
                return 0.0
            
            return (hawkish_count - dovish_count) / total_keywords
            
        except Exception as e:
            logger.error(f"Error analyzing hawkish/dovish tone: {str(e)}")
            return 0.0
    
    def _predict_policy_changes(self, communication: Dict[str, Any]) -> float:
        """Predict probability of policy changes"""
        try:
            content = communication.get('content', '').lower()
            change_keywords = ['consider', 'evaluate', 'review', 'assess', 'monitor']
            
            change_signals = sum(1 for keyword in change_keywords if keyword in content)
            return min(change_signals * 0.2, 1.0)  # Max 1.0 probability
            
        except Exception as e:
            logger.error(f"Error predicting policy changes: {str(e)}")
            return 0.0
    
    def _update_policy_expectations(self, currency: str, hawkish_dovish: float, change_prob: float):
        """Update policy expectations for currency"""
        try:
            if currency not in self.sentiment_data:
                self.sentiment_data[currency] = []
            
            policy_update = {
                'timestamp': datetime.now(),
                'hawkish_dovish_score': hawkish_dovish,
                'policy_change_probability': change_prob,
                'type': 'central_bank_communication'
            }
            
            self.sentiment_data[currency].append(policy_update)
            
        except Exception as e:
            logger.error(f"Error updating policy expectations: {str(e)}")
    
    async def _process_economic_calendar(self):
        """Process upcoming economic calendar events"""
        try:
            upcoming_events = await self._get_upcoming_events()
            
            for event_data in upcoming_events:
                event = self._create_economic_event(event_data)
                if event:
                    self.economic_events.append(event)
                    
                    # Generate volatility prediction
                    volatility_pred = self._predict_event_volatility(event)
                    self.volatility_predictions[event.currency] = volatility_pred
            
            # Keep only future events
            now = datetime.now()
            self.economic_events = [e for e in self.economic_events if e.timestamp > now]
            
        except Exception as e:
            logger.error(f"Error processing economic calendar: {str(e)}")
    
    async def _get_upcoming_events(self) -> List[Dict[str, Any]]:
        """Get upcoming economic events from ForexFactory calendar"""
        try:
            async with aiohttp.ClientSession() as session:
                # ForexFactory calendar API (free)
                url = "https://www.forexfactory.com/calendar.php"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        events = self._parse_forexfactory_calendar(content)
                        
                        logger.info(f"Fetched {len(events)} upcoming events from ForexFactory")
                        return events
                    else:
                        logger.warning(f"ForexFactory calendar returned status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching upcoming events: {str(e)}")
            return []
    
    def _parse_forexfactory_calendar(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse ForexFactory calendar HTML"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            events = []
            
            # Find calendar events table
            calendar_rows = soup.find_all('tr', class_='calendar_row')
            
            for row in calendar_rows[:20]:  # Limit to next 20 events
                try:
                    # Extract event data from row
                    time_cell = row.find('td', class_='calendar__time')
                    currency_cell = row.find('td', class_='calendar__currency')
                    event_cell = row.find('td', class_='calendar__event')
                    impact_cell = row.find('td', class_='calendar__impact')
                    
                    if time_cell and currency_cell and event_cell:
                        # Parse impact level from stars
                        impact_spans = impact_cell.find_all('span', class_='gmt') if impact_cell else []
                        impact_level = 'HIGH' if len(impact_spans) >= 3 else 'MEDIUM' if len(impact_spans) == 2 else 'LOW'
                        
                        event = {
                            'timestamp': datetime.now() + timedelta(hours=2),  # Placeholder
                            'currency': currency_cell.text.strip() if currency_cell else 'USD',
                            'event_name': event_cell.text.strip() if event_cell else 'Economic Event',
                            'impact_level': impact_level,
                            'forecast': None,
                            'previous': None
                        }
                        events.append(event)
                        
                except Exception as e:
                    logger.debug(f"Error parsing calendar row: {str(e)}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Error parsing ForexFactory calendar: {str(e)}")
            return []
    
    def _create_economic_event(self, event_data: Dict[str, Any]) -> Optional[EconomicEvent]:
        """Create EconomicEvent from raw data"""
        try:
            return EconomicEvent(
                timestamp=event_data['timestamp'],
                currency=event_data['currency'],
                event_name=event_data['event_name'],
                impact_level=ImpactLevel(event_data['impact_level'].lower()),
                actual_value=event_data.get('actual'),
                forecast_value=event_data.get('forecast'),
                previous_value=event_data.get('previous'),
                sentiment_score=0.0,  # Will be calculated
                volatility_prediction=0.0  # Will be calculated
            )
        except Exception as e:
            logger.error(f"Error creating economic event: {str(e)}")
            return None
    
    def _predict_event_volatility(self, event: EconomicEvent) -> float:
        """Predict volatility impact of economic event"""
        try:
            base_volatility = {
                ImpactLevel.LOW: 0.5,
                ImpactLevel.MEDIUM: 1.0,
                ImpactLevel.HIGH: 2.0,
                ImpactLevel.CRITICAL: 3.0
            }.get(event.impact_level, 1.0)
            
            # Adjust based on forecast vs previous deviation
            if event.forecast_value and event.previous_value and event.previous_value != 0:
                deviation = abs(event.forecast_value - event.previous_value) / abs(event.previous_value)
                base_volatility *= (1 + deviation)
            
            return min(base_volatility, 5.0)  # Cap at 5x normal volatility
            
        except Exception as e:
            logger.error(f"Error predicting event volatility: {str(e)}")
            return 1.0
    
    async def _predict_event_impact(self):
        """Predict overall market impact of upcoming events"""
        try:
            now = datetime.now()
            next_24h = now + timedelta(hours=24)
            
            upcoming_events = [e for e in self.economic_events if now < e.timestamp < next_24h]
            
            if upcoming_events:
                # Calculate aggregate impact by currency
                currency_impacts = {}
                for event in upcoming_events:
                    if event.currency not in currency_impacts:
                        currency_impacts[event.currency] = 0.0
                    currency_impacts[event.currency] += event.volatility_prediction
                
                # Log high impact predictions
                for currency, impact in currency_impacts.items():
                    if impact > self.impact_thresholds['volatility_spike_threshold']:
                        logger.warning(f"HIGH VOLATILITY PREDICTED: {currency} - impact score: {impact:.2f}")
            
        except Exception as e:
            logger.error(f"Error predicting event impact: {str(e)}")
    
    # Public API methods
    def get_economic_outlook(self, currency: str = None) -> Dict[str, Any]:
        """Get economic outlook for currency"""
        try:
            if currency:
                sentiment_data = self.sentiment_data.get(currency, [])
                recent_sentiment = sentiment_data[-10:] if sentiment_data else []
                
                avg_sentiment = np.mean([s.get('sentiment', 0) for s in recent_sentiment]) if recent_sentiment else 0
                
                return {
                    'currency': currency,
                    'avg_sentiment': float(avg_sentiment),
                    'sentiment_trend': 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral',
                    'upcoming_events': len([e for e in self.economic_events if e.currency == currency]),
                    'volatility_prediction': self.volatility_predictions.get(currency, 1.0)
                }
            else:
                return {
                    'monitored_currencies': list(self.sentiment_data.keys()),
                    'total_events': len(self.economic_events),
                    'high_impact_events': len([e for e in self.economic_events if e.impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]])
                }
                
        except Exception as e:
            logger.error(f"Error getting economic outlook: {str(e)}")
            return {}
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get upcoming economic events"""
        try:
            cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
            return [e for e in self.economic_events if e.timestamp <= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error getting upcoming events: {str(e)}")
            return []
    
    def get_volatility_forecast(self, currency: str) -> Dict[str, Any]:
        """Get volatility forecast for currency"""
        try:
            base_volatility = self.volatility_predictions.get(currency, 1.0)
            sentiment_data = self.sentiment_data.get(currency, [])
            
            # Calculate sentiment-adjusted volatility
            if sentiment_data:
                recent_sentiment = sentiment_data[-5:]
                sentiment_volatility = np.std([abs(s.get('sentiment', 0)) for s in recent_sentiment])
                adjusted_volatility = base_volatility * (1 + sentiment_volatility)
            else:
                adjusted_volatility = base_volatility
            
            return {
                'currency': currency,
                'base_volatility_multiplier': base_volatility,
                'sentiment_adjusted_multiplier': adjusted_volatility,
                'forecast_confidence': 0.7,  # Static for now
                'time_horizon': '24_hours'
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility forecast: {str(e)}")
            return {}
    
    async def shutdown(self):
        """Shutdown the economic intelligence agent"""
        try:
            self.is_initialized = False
            if hasattr(self, 'news_task'):
                self.news_task.cancel()
            if hasattr(self, 'event_task'):
                self.event_task.cancel()
            logger.info("Economic Intelligence Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Economic Intelligence Agent: {str(e)}")
