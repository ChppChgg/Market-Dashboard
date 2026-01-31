"""
Sentiment.py - LLM-Powered Market Sentiment Analysis
Uses Gemini 1.5 Flash to analyze market news headlines and provide sentiment ratings.

Features:
- Fetches recent news from Google News RSS
- Analyzes headlines using Gemini for financial context
- Returns structured sentiment with sources
"""

import os
import json
import time
import feedparser
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Gemini SDK (new package)
from google import genai
from google.genai import types


# =============================================================================
# CONFIGURATION
# =============================================================================

# API Key (hardcoded for now - move to .env for production)
GEMINI_API_KEY = "AIzaSyDJ-kYaNWQ0az1-dZOKT2K9R7dok7IVlm8"

# Model settings
GEMINI_MODEL = "gemini-2.0-flash-lite"  # Lite version, better for free tier
MAX_HEADLINES = 10
NEWS_CACHE_HOURS = 2  # How long to cache news results

# Rate limiting settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5  # Wait longer than the 59s suggested by API
REQUEST_DELAY_SECONDS = 2  # Delay between requests to avoid rate limits

# Sentiment ratings
SENTIMENT_RATINGS = {
    "Good": {"color": "#00ff88", "emoji": "🟢"},
    "Mixed": {"color": "#ffc107", "emoji": "🟡"},
    "Bad": {"color": "#ff4444", "emoji": "🔴"},
    "Unknown": {"color": "#888888", "emoji": "⚪"}
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NewsHeadline:
    """Single news headline with metadata."""
    title: str
    source: str
    link: str
    published: str
    

@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""
    overall_sentiment: str  # "Good", "Mixed", "Bad"
    confidence: float       # 0.0 - 1.0
    summary: str           # 2-3 sentence summary
    key_themes: List[str]  # Main themes identified
    bullish_signals: List[str]
    bearish_signals: List[str]
    headlines: List[Dict]  # Original headlines with sources
    analyzed_at: str       # Timestamp
    ticker: str            # What was analyzed
    error: Optional[str] = None  # Error message if failed


# =============================================================================
# NEWS FETCHING
# =============================================================================

def fetch_news_headlines(query: str, num_articles: int = MAX_HEADLINES) -> List[NewsHeadline]:
    """
    Fetch news headlines from Google News RSS.
    
    Args:
        query: Search query (e.g., "AAPL stock", "gold price")
        num_articles: Number of articles to fetch
    
    Returns:
        List of NewsHeadline objects
    """
    try:
        # Google News RSS URL
        rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        # Parse RSS feed
        feed = feedparser.parse(rss_url)
        
        headlines = []
        for entry in feed.entries[:num_articles]:
            # Extract source from title (Google News format: "Title - Source")
            title_parts = entry.title.rsplit(" - ", 1)
            title = title_parts[0] if len(title_parts) > 1 else entry.title
            source = title_parts[1] if len(title_parts) > 1 else "Unknown"
            
            headlines.append(NewsHeadline(
                title=title,
                source=source,
                link=entry.link,
                published=entry.get("published", "Unknown date")
            ))
        
        return headlines
        
    except Exception as e:
        print(f"[Sentiment] Error fetching news: {e}")
        return []


def build_query_for_ticker(ticker: str) -> str:
    """
    Build a search query optimized for financial news.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Optimized search query
    """
    # Common ticker mappings for better results
    ticker_names = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq",
        "GC=F": "gold price",
        "SI=F": "silver price",
        "CL=F": "oil price crude",
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
    }
    
    if ticker in ticker_names:
        return f"{ticker_names[ticker]} market news"
    else:
        return f"{ticker} stock market news"


# =============================================================================
# GEMINI ANALYSIS
# =============================================================================

# Global client instance
_gemini_client = None

def get_gemini_client(api_key: str = None):
    """Get or create Gemini client."""
    global _gemini_client
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Set it as an environment variable or pass it directly.")
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def configure_gemini(api_key: str = None):
    """Configure Gemini API with the provided key (for backwards compatibility)."""
    get_gemini_client(api_key)


def analyze_with_gemini(headlines: List[NewsHeadline], ticker: str) -> SentimentResult:
    """
    Analyze headlines using Gemini 1.5 Flash.
    
    Args:
        headlines: List of NewsHeadline objects
        ticker: Ticker being analyzed
    
    Returns:
        SentimentResult with analysis
    """
    if not headlines:
        return SentimentResult(
            overall_sentiment="Unknown",
            confidence=0.0,
            summary="No news headlines found for analysis.",
            key_themes=[],
            bullish_signals=[],
            bearish_signals=[],
            headlines=[],
            analyzed_at=datetime.now().isoformat(),
            ticker=ticker,
            error="No headlines to analyze"
        )
    
    # Format headlines for the prompt
    formatted_headlines = "\n".join([
        f"{i+1}. \"{h.title}\" - {h.source}"
        for i, h in enumerate(headlines)
    ])
    
    # Construct the analysis prompt
    prompt = f"""You are a financial market analyst. Analyze these {len(headlines)} recent news headlines about {ticker}:

{formatted_headlines}

Based on these headlines, provide a sentiment analysis. Consider:
- Overall market direction signals (bullish/bearish)
- Analyst opinions and price targets
- Macroeconomic factors mentioned
- Company-specific news (earnings, products, management)
- Market conditions and trends

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
    "overall_sentiment": "Good" or "Mixed" or "Bad",
    "confidence": 0.0 to 1.0,
    "summary": "2-3 sentence summary of the overall market sentiment",
    "key_themes": ["theme1", "theme2", "theme3"],
    "bullish_signals": ["positive signal 1", "positive signal 2"],
    "bearish_signals": ["negative signal 1", "negative signal 2"],
    "headline_sentiments": ["Good", "Bad", "Neutral", ...]
}}

Rules:
- "Good" = mostly positive news, bullish outlook
- "Mixed" = balanced or conflicting signals  
- "Bad" = mostly negative news, bearish outlook
- Keep summary concise but informative
- List 2-4 key themes
- If no bullish/bearish signals, use empty array []
- headline_sentiments: Array of sentiment for EACH headline in order ("Good", "Neutral", or "Bad")
"""

    try:
        # Get client and generate response
        client = get_gemini_client()
        
        # Retry loop for rate limiting
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # Small delay before each request to avoid hitting rate limits
                if attempt > 0:
                    wait_time = RETRY_DELAY_SECONDS
                    print(f"[Sentiment] Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                else:
                    time.sleep(REQUEST_DELAY_SECONDS)  # Small initial delay
                
                # Generate response using new SDK
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        continue  # Will retry after delay
                    else:
                        raise  # Max retries exceeded
                else:
                    raise  # Non-rate-limit error, don't retry
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Clean up response if it has markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        # Build headline list with links and individual sentiments
        headline_sentiments = result.get("headline_sentiments", [])
        headline_dicts = []
        for i, h in enumerate(headlines):
            sentiment = headline_sentiments[i] if i < len(headline_sentiments) else "Neutral"
            headline_dicts.append({
                "title": h.title,
                "source": h.source,
                "link": h.link,
                "published": h.published,
                "sentiment": sentiment
            })
        
        return SentimentResult(
            overall_sentiment=result.get("overall_sentiment", "Mixed"),
            confidence=float(result.get("confidence", 0.5)),
            summary=result.get("summary", "Analysis unavailable."),
            key_themes=result.get("key_themes", []),
            bullish_signals=result.get("bullish_signals", []),
            bearish_signals=result.get("bearish_signals", []),
            headlines=headline_dicts,
            analyzed_at=datetime.now().isoformat(),
            ticker=ticker
        )
        
    except json.JSONDecodeError as e:
        print(f"[Sentiment] JSON parse error: {e}")
        print(f"[Sentiment] Raw response: {response.text[:500]}")
        return SentimentResult(
            overall_sentiment="Unknown",
            confidence=0.0,
            summary="Failed to parse analysis response.",
            key_themes=[],
            bullish_signals=[],
            bearish_signals=[],
            headlines=[asdict(h) for h in headlines],
            analyzed_at=datetime.now().isoformat(),
            ticker=ticker,
            error=f"JSON parse error: {str(e)}"
        )
        
    except Exception as e:
        print(f"[Sentiment] Gemini API error: {e}")
        return SentimentResult(
            overall_sentiment="Unknown",
            confidence=0.0,
            summary=f"Analysis failed: {str(e)}",
            key_themes=[],
            bullish_signals=[],
            bearish_signals=[],
            headlines=[asdict(h) for h in headlines],
            analyzed_at=datetime.now().isoformat(),
            ticker=ticker,
            error=str(e)
        )


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_sentiment(ticker: str, api_key: str = None, num_headlines: int = MAX_HEADLINES) -> SentimentResult:
    """
    Main function to analyze market sentiment for a ticker.
    
    Args:
        ticker: Stock/asset ticker symbol
        api_key: Gemini API key (optional if set in environment)
        num_headlines: Number of headlines to analyze
    
    Returns:
        SentimentResult with complete analysis
    
    Example:
        result = analyze_sentiment("AAPL")
        print(f"Sentiment: {result.overall_sentiment}")
        print(f"Summary: {result.summary}")
    """
    # Configure API
    configure_gemini(api_key)
    
    # Build search query
    query = build_query_for_ticker(ticker)
    print(f"[Sentiment] Searching for: {query}")
    
    # Fetch headlines
    headlines = fetch_news_headlines(query, num_headlines)
    print(f"[Sentiment] Found {len(headlines)} headlines")
    
    # Analyze with Gemini
    result = analyze_with_gemini(headlines, ticker)
    
    return result


def get_sentiment_color(sentiment: str) -> str:
    """Get the color for a sentiment rating."""
    return SENTIMENT_RATINGS.get(sentiment, SENTIMENT_RATINGS["Unknown"])["color"]


def get_sentiment_emoji(sentiment: str) -> str:
    """Get the emoji for a sentiment rating."""
    return SENTIMENT_RATINGS.get(sentiment, SENTIMENT_RATINGS["Unknown"])["emoji"]


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Set it with: set GEMINI_API_KEY=your_key_here (Windows)")
        print("Or: export GEMINI_API_KEY=your_key_here (Linux/Mac)")
        sys.exit(1)
    
    # Test with a ticker
    test_ticker = "SPY"
    print(f"\n{'='*60}")
    print(f"Testing Sentiment Analysis for: {test_ticker}")
    print(f"{'='*60}\n")
    
    result = analyze_sentiment(test_ticker)
    
    # Display results
    emoji = get_sentiment_emoji(result.overall_sentiment)
    print(f"\n{emoji} Overall Sentiment: {result.overall_sentiment}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nSummary: {result.summary}")
    
    if result.key_themes:
        print(f"\nKey Themes:")
        for theme in result.key_themes:
            print(f"  • {theme}")
    
    if result.bullish_signals:
        print(f"\n📈 Bullish Signals:")
        for signal in result.bullish_signals:
            print(f"  • {signal}")
    
    if result.bearish_signals:
        print(f"\n📉 Bearish Signals:")
        for signal in result.bearish_signals:
            print(f"  • {signal}")
    
    print(f"\nSources ({len(result.headlines)} articles):")
    for i, h in enumerate(result.headlines, 1):
        print(f"  {i}. {h['title'][:60]}... - {h['source']}")
    
    if result.error:
        print(f"\n⚠️ Error: {result.error}")
