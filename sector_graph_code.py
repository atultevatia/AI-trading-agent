import os
import operator
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import yfinance as yf
import pandas as pd
from news_engine import news_engine

# Load environment variables
load_dotenv()

# --- Configuration ---
CAPITAL = 10000
RISK_PER_TRADE_PCT = 0.01  # 1%
MAX_RISK_INR = CAPITAL * RISK_PER_TRADE_PCT

import requests
from io import StringIO

# --- Constants & Prompts ---
# Official NSE Index Constituent CSVs
SECTOR_URLS = {
    "AUTO": "https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",
    "IT": "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "BANK": "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "PHARMA": "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "FMCG": "https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv"
}

# Fallback in case NSE URLs are down
FALLBACK_MAPPING = {
    "AI": ["TATAELXSI.NS", "PERSISTENT.NS", "OFSS.NS", "CYIENT.NS", "HAPPSTMNDS.NS", "KPITTECH.NS", "ZENTEC.NS"],
    "AUTO": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "ASHOKLEY.NS"],
    "BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "TORNTPHARM.NS"],
    "FMCG": ["HUL.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS"]
}


ANALYST_PROMPT = """You are a Conservative Quantitative Trader using a Tiered Decision Framework.
Input: Price Data (OHLCV), Technical Indicators, and News.

Task:
1. Calculate Technical Score (0-1) based on Trend, Momentum, and Support.
2. Calculate Sentiment Score (0-1) based on News.
3. Assign a Tier based on STRICT criteria:
   - **STRONG_BUY**: Tech Score > 0.65 AND Sentiment > 0.3
   - **WATCHLIST**: Tech Score > 0.55 AND Sentiment > 0.1
   - **MONITOR**: Tech Score > 0.5
   - **SKIP**: Anything else.

Output: JSON with keys:
{
  "tech_score": float,
  "sentiment_score": float,
  "tier": "STRONG_BUY" | "WATCHLIST" | "MONITOR" | "SKIP",
  "confidence": float (0-1),
  "reasoning": "concise explanation"
}
"""

# --- State Definitions ---

class StockResult(TypedDict):
    ticker: str
    price: float
    tech_score: float
    sentiment_score: float
    tier: str
    confidence: float
    entry: float
    stop: float
    target: float
    position_size: int
    reasoning: str

class OverallState(TypedDict):
    sector: str
    tickers: List[str]
    results: List[StockResult]
    final_ranking: List[StockResult]

# --- Helper Functions ---
def fetch_nse_constituents(sector_key: str):
    url = SECTOR_URLS.get(sector_key)
    if not url:
        return FALLBACK_MAPPING.get(sector_key, [])
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            # NSE CSVs have a 'Symbol' column. We append '.NS' for yfinance compatibility.
            return [f"{sym}.NS" for sym in df['Symbol'].tolist()]
        else:
            print(f"Warning: Failed to fetch NSE data for {sector_key}. Using fallback.")
            return FALLBACK_MAPPING.get(sector_key, [])
    except Exception as e:
        print(f"Error fetching live NSE data for {sector_key}: {e}")
        return FALLBACK_MAPPING.get(sector_key, [])

def get_technical_indicators(ticker: str):

    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            return None
        
        # Simple TA
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        return {
            "price": round(latest['Close'].item(), 2),
            "sma_50": round(latest['SMA_50'].item(), 2) if not pd.isna(latest['SMA_50']) else 0,
            "sma_200": round(latest['SMA_200'].item(), 2) if not pd.isna(latest['SMA_200']) else 0,
            "rsi": round(latest['RSI'].item(), 2) if not pd.isna(latest['RSI']) else 50
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# --- Nodes ---

def sector_loader_node(state: OverallState):
    sector = state['sector'].upper()
    # Map AI to IT for NSE constituents
    fetch_key = "IT" if sector == "AI" else sector
    tickers = fetch_nse_constituents(fetch_key)
    print(f"--- Loaded {len(tickers)} stocks for {sector} ---")
    return {"tickers": tickers, "results": []}


def analyst_node(state: OverallState):
    results = []
    tickers = state['tickers']
    if not tickers:
        return {"results": []}

    print(f"--- Batch Fetching Data for {len(tickers)} stocks ---")
    try:
        # Download all at once
        data = yf.download(tickers, period="7mo", interval="1d", progress=False, group_by='ticker')
        if data.empty:
            print("No data fetched from Yahoo Finance.")
            return {"results": []}
    except Exception as e:
        print(f"Batch Download Error: {e}")
        return {"results": []}

    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    
    for ticker in tickers:
        try:
            print(f"Analyzing {ticker}...")
            
            # Extract ticker data safely from multi-index dataframe
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]:
                    print(f"No data for {ticker} in batch results.")
                    continue
                df = data[ticker].copy().dropna()
            else:
                df = data.copy().dropna()
                
            if df.empty or len(df) < 50: # Lowered threshold slightly for variety
                print(f"Skipping {ticker}: Insufficient data ({len(df)} rows).")
                continue

            # Ensure we have the necessary columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {ticker}: Missing columns.")
                continue

            # Calculate Technicals locally
            # Use .iloc[:, 0] in case of duplicate column names to force Series
            close_ser = df['Close']
            if isinstance(close_ser, pd.DataFrame):
                close_ser = close_ser.iloc[:, 0]

            df['SMA_50'] = close_ser.rolling(window=50).mean()
            df['SMA_200'] = close_ser.rolling(window=200).mean()
            
            delta = close_ser.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            current_price = float(latest['Close'])
            
            ta_data = {
                "price": round(current_price, 2),
                "sma_50": round(float(latest['SMA_50']), 2) if not pd.isna(latest['SMA_50']) else 0,
                "sma_200": round(float(latest['SMA_200']), 2) if not pd.isna(latest['SMA_200']) else 0,
                "rsi": round(float(latest['RSI']), 2) if not pd.isna(latest['RSI']) else 50
            }

            # Fetch Real-Time News
            news_items = news_engine.get_stock_news(ticker)
            if news_items:
                news_snippet = "\n".join([f"- {n['title']} ({n['published']})" for n in news_items[:5]])
            else:
                news_snippet = "No recent major news headlines found."
            
            # LLM Analysis
            msg = f"""Ticker: {ticker}
            Price: {ta_data['price']}
            SMA50: {ta_data['sma_50']}
            SMA200: {ta_data['sma_200']}
            RSI: {ta_data['rsi']}
            
            Recent News Headlines:
            {news_snippet}
            """
            
            response = llm.invoke([
                SystemMessage(content=ANALYST_PROMPT),
                HumanMessage(content=msg)
            ])
            analysis = json.loads(response.content)
            
            tier = analysis.get('tier', 'SKIP')
            if tier == "SKIP":
                continue

            # Risk Model
            stop_loss = round(current_price * 0.95, 2)
            risk_per_share = current_price - stop_loss
            pos_size = int(MAX_RISK_INR / risk_per_share) if risk_per_share > 0 else 0
            
            result: StockResult = {
                "ticker": ticker,
                "price": ta_data['price'],
                "tech_score": analysis.get('tech_score', 0),
                "sentiment_score": analysis.get('sentiment_score', 0),
                "tier": tier,
                "confidence": analysis.get('confidence', 0),
                "entry": ta_data['price'],
                "stop": stop_loss,
                "target": round(current_price * 1.1, 2),
                "position_size": pos_size,
                "reasoning": analysis.get('reasoning', "N/A")
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return {"results": results}


def ranker_node(state: OverallState):
    print("--- Ranking Candidates ---")
    results = state['results']
    
    # Priority: STRONG_BUY > WATCHLIST > MONITOR
    tier_priority = {"STRONG_BUY": 3, "WATCHLIST": 2, "MONITOR": 1}
    
    # Sort by Tier Priority DESC, then Confidence DESC
    ranked = sorted(results, key=lambda x: (tier_priority.get(x['tier'], 0), x['confidence']), reverse=True)
    
    return {"final_ranking": ranked}

# --- Graph Wiring ---
def create_sector_graph():
    workflow = StateGraph(OverallState)
    
    workflow.add_node("SectorLoader", sector_loader_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Ranker", ranker_node)
    
    workflow.set_entry_point("SectorLoader")
    
    workflow.add_edge("SectorLoader", "Analyst")
    workflow.add_edge("Analyst", "Ranker")
    workflow.add_edge("Ranker", END)
    
    return workflow.compile()

app = create_sector_graph()
