import os
import operator
import json
import time
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from news_engine import news_engine

# Load environment variables
load_dotenv()

# --- Configuration ---
CAPITAL = 10000

# --- Constants & Prompts ---
SECTOR_URLS = {
    "AUTO": "https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",
    "IT": "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "BANK": "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "PHARMA": "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "FMCG": "https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv"
}

FALLBACK_MAPPING = {
    "AI": ["TATAELXSI.NS", "PERSISTENT.NS", "OFSS.NS", "CYIENT.NS", "HAPPSTMNDS.NS", "KPITTECH.NS", "ZENTEC.NS"],
    "AUTO": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "ASHOKLEY.NS"],
    "BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "TORNTPHARM.NS"],
    "FMCG": ["HUL.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS"]
}

ANALYST_AGENT_PROMPT = """You are a Lead Quantitative Research Analyst. 
Your task is to synthesize technical data and news to identify high-probability swing trades.
Move beyond simple rules; reason about the convergence of price action and sentiment.

Output JSON:
{
    "thesis": "detailed reasoning",
    "conviction": 0-100,
    "entry": float,
    "target": float,
    "stop_loss": float
}
"""

RISK_MANAGER_PROMPT = """You are a Conservative Risk Officer. Your job is to find reasons NOT to take the trade pitched by the Analyst.
Critique the entry, stop loss, and news sentiment.

Output JSON:
{
    "status": "APPROVED" | "REJECTED",
    "risk_criticism": "reasoning for selection",
    "adjusted_stop": float
}
"""

PORTFOLIO_MANAGER_PROMPT = """You are a Fund Manager. Total budget: ₹10,000.
Allocate capital across approved trades based on conviction and risk. Max 40% per stock.

Output JSON:
{
    "allocations": [
        {"ticker": "string", "shares": int, "weight_pct": float, "reason": "why"}
    ],
    "remaining_cash": float
}
"""

# --- Caching ---

class TradeCache:
    def __init__(self, ttl_seconds=7200): # 2 hour cache
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, ticker: str):
        if ticker in self.cache:
            entry, timestamp = self.cache[ticker]
            if time.time() - timestamp < self.ttl:
                return entry
            del self.cache[ticker]
        return None

    def set(self, ticker: str, analysis: dict):
        self.cache[ticker] = (analysis, time.time())

trade_cache = TradeCache()
sector_cache = {} # Cache for tickers per sector

# --- State Definitions ---

class StockAnalysis(TypedDict):
    ticker: str
    price: float
    thesis: str
    conviction: int
    entry: float
    target: float
    stop_loss: float
    risk_status: str
    risk_criticism: str
    adjusted_stop: float

class OverallState(TypedDict):
    sector: str
    tickers: List[str]
    analyses: List[StockAnalysis]
    portfolio: List[Dict[str, Any]]
    remaining_cash: float

# --- Helper Functions ---

def fetch_nse_constituents(sector_key: str):
    url = SECTOR_URLS.get(sector_key)
    if not url: return FALLBACK_MAPPING.get(sector_key, [])
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return [f"{sym}.NS" for sym in df['Symbol'].tolist()]
        return FALLBACK_MAPPING.get(sector_key, [])
    except: return FALLBACK_MAPPING.get(sector_key, [])

# --- Nodes ---

def sector_loader_node(state: OverallState):
    sector = state['sector'].upper()
    if sector in sector_cache:
        print(f"Using cached constituents for {sector}")
        tickers = sector_cache[sector]
    else:
        fetch_key = "IT" if sector == "AI" else sector
        tickers = fetch_nse_constituents(fetch_key)
        sector_cache[sector] = tickers
    return {"tickers": tickers, "analyses": [], "portfolio": [], "remaining_cash": CAPITAL}

def research_pipeline(ticker: str, ticker_data: pd.DataFrame):
    """Combines Analysis and Risk Review into a single thread for speed."""
    # 1. Analyst Stage
    cached = trade_cache.get(ticker)
    analysis = None
    if cached:
        analysis = cached
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
        try:
            close_ser = ticker_data['Close']
            if isinstance(close_ser, pd.DataFrame): close_ser = close_ser.iloc[:, 0]
            sma50 = close_ser.rolling(50).mean().iloc[-1]
            sma200 = close_ser.rolling(200).mean().iloc[-1]
            news = news_engine.get_stock_news(ticker)
            news_txt = "\n".join([n['title'] for n in news[:3]])
            msg = f"Ticker: {ticker}\nPrice: {close_ser.iloc[-1]}\nSMA50: {sma50}\nSMA200: {sma200}\nNews: {news_txt}"
            response = llm.invoke([SystemMessage(content=ANALYST_AGENT_PROMPT), HumanMessage(content=msg)])
            res = json.loads(response.content)
            analysis = {
                "ticker": ticker,
                "price": float(close_ser.iloc[-1]),
                "thesis": res['thesis'],
                "conviction": res['conviction'],
                "entry": res['entry'],
                "target": res['target'],
                "stop_loss": res['stop_loss']
            }
            # Cache the raw analysis first
            trade_cache.set(ticker, analysis)
        except Exception as e:
            print(f"Analyst Error on {ticker}: {e}")
            return None

    # 2. Risk Manager Stage (immediately following logic)
    # Even if cached, we might need a fresh risk review or we can cache the vetted one
    # For now, let's run risk review as part of the same thread
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
        msg = f"Trade Pitch for {analysis['ticker']}:\n{json.dumps(analysis, indent=2)}"
        response = llm.invoke([SystemMessage(content=RISK_MANAGER_PROMPT), HumanMessage(content=msg)])
        res = json.loads(response.content)

        if res['status'] == "APPROVED":
            analysis.update({
                "risk_status": "APPROVED",
                "risk_criticism": res['risk_criticism'],
                "adjusted_stop": res['adjusted_stop']
            })
        else:
            analysis.update({
                "risk_status": "REJECTED",
                "risk_criticism": res['risk_criticism'],
                "adjusted_stop": analysis['stop_loss']
            })
        return analysis
    except Exception as e:
        print(f"Risk Review Error on {analysis['ticker']}: {e}")
        return analysis # Return partial analysis if risk check fails

def researcher_node(state: OverallState):
    """High-speed parallel research pipeline."""
    tickers = state['tickers']
    if not tickers: return {"analyses": []}

    print(f"--- Researcher: High-Concurrency Pipe ({len(tickers)} stocks) ---")
    data = yf.download(tickers, period="7mo", interval="1d", progress=False, group_by='ticker')
    
    tasks = []
    # Boosted workers to 20 for maximum overlap of network/LLM calls
    with ThreadPoolExecutor(max_workers=20) as executor:
        for ticker in tickers:
            ticker_df = data[ticker].copy().dropna() if len(tickers) > 1 else data.copy().dropna()
            if len(ticker_df) >= 50:
                tasks.append(executor.submit(research_pipeline, ticker, ticker_df))
    
    results = [task.result() for task in tasks if task.result() is not None]
    return {"analyses": results}


def portfolio_manager_node(state: OverallState):
    print("--- Portfolio Manager: Allocating Capital ---")
    approved_trades = [a for a in state['analyses'] if a.get('risk_status') == "APPROVED"]
    if not approved_trades: return {"portfolio": [], "remaining_cash": CAPITAL}

    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    msg = f"Approved Trades:\n{json.dumps(approved_trades, indent=2)}\nMax Capital: ₹{CAPITAL}"

    
    try:
        response = llm.invoke([SystemMessage(content=PORTFOLIO_MANAGER_PROMPT), HumanMessage(content=msg)])
        res = json.loads(response.content)
        return {"portfolio": res['allocations'], "remaining_cash": res.get('remaining_cash', 0)}
    except Exception as e:
        print(f"Portfolio Manager Error: {e}")
        return {"portfolio": []}

# --- Graph ---

def create_agent_graph():
    workflow = StateGraph(OverallState)
    workflow.add_node("Loader", sector_loader_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("PortfolioManager", portfolio_manager_node)

    workflow.set_entry_point("Loader")
    workflow.add_edge("Loader", "Researcher")
    workflow.add_edge("Researcher", "PortfolioManager")
    workflow.add_edge("PortfolioManager", END)

    return workflow.compile()

app = create_agent_graph()

